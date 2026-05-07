#include <windows.h>
#include <wbemidl.h>
#include <comdef.h>

#include <openssl/bn.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/rsa.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <numeric>
#include <string>
#include <vector>

struct BatteryInfo {
    int battery_pct;
    int is_charging;
};

struct TelemetryInfo {
    int battery_pct;
    double cpu_temp_c;
    int is_charging;
};

struct HysteresisThresholds {
    int switch_to_multiprime_battery_pct = 30;
    double switch_to_multiprime_temp_c = 65.0;
    int switch_to_standard_battery_pct = 40;
    double switch_to_standard_temp_c = 55.0;
    double switch_to_standard_hold_s = 120.0;
};

enum class CryptoMode {
    Standard = 0,
    MultiPrime = 1
};

struct KeyContext {
    std::string scenario_name;
    EVP_PKEY* key = nullptr;
    EVP_PKEY_CTX* decrypt_ctx = nullptr;
    std::vector<unsigned char> ciphertext;
};

struct RuntimeOptions {
    std::string out_csv = "rsa_benchmark.csv";
    std::string micro_out_csv = "rsa_microbenchmark.csv";
    int iterations = 300;
    int key_bits = 2048;
    int micro_iterations = 5000;
    int warmup_iterations = 500;
    double power_watts = 5.0;
    bool run_microbenchmark = false;
    bool print_key_diagnostics = true;
    HysteresisThresholds thresholds{};
    std::optional<int> mock_battery_pct;
    std::optional<double> mock_temp_c;
    std::optional<int> mock_charging;
};

static bool g_print_key_diagnostics = true;

BatteryInfo get_battery_info() {
    SYSTEM_POWER_STATUS sps{};
    if (!GetSystemPowerStatus(&sps)) return {-1, 0};
    int pct = (sps.BatteryLifePercent == 255) ? -1 : (int)sps.BatteryLifePercent;
    int charging = (sps.ACLineStatus == 1) ? 1 : 0;
    return {pct, charging};
}

double get_system_temperature_c() {
    HRESULT hres = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    bool uninit = SUCCEEDED(hres);
    if (FAILED(hres) && hres != RPC_E_CHANGED_MODE) return std::numeric_limits<double>::quiet_NaN();

    hres = CoInitializeSecurity(nullptr, -1, nullptr, nullptr,
                                RPC_C_AUTHN_LEVEL_DEFAULT, RPC_C_IMP_LEVEL_IMPERSONATE,
                                nullptr, EOAC_NONE, nullptr);
    if (FAILED(hres) && hres != RPC_E_TOO_LATE) {
        if (uninit) CoUninitialize();
        return std::numeric_limits<double>::quiet_NaN();
    }

    IWbemLocator* pLoc = nullptr;
    hres = CoCreateInstance(CLSID_WbemLocator, nullptr, CLSCTX_INPROC_SERVER,
                            IID_IWbemLocator, (LPVOID*)&pLoc);
    if (FAILED(hres) || !pLoc) {
        if (uninit) CoUninitialize();
        return std::numeric_limits<double>::quiet_NaN();
    }

    IWbemServices* pSvc = nullptr;
    hres = pLoc->ConnectServer(_bstr_t(L"ROOT\\WMI"), nullptr, nullptr, nullptr, 0, nullptr, nullptr, &pSvc);
    if (FAILED(hres) || !pSvc) {
        pLoc->Release();
        if (uninit) CoUninitialize();
        return std::numeric_limits<double>::quiet_NaN();
    }

    hres = CoSetProxyBlanket(pSvc, RPC_C_AUTHN_WINNT, RPC_C_AUTHZ_NONE, nullptr,
                             RPC_C_AUTHN_LEVEL_CALL, RPC_C_IMP_LEVEL_IMPERSONATE, nullptr, EOAC_NONE);
    if (FAILED(hres)) {
        pSvc->Release();
        pLoc->Release();
        if (uninit) CoUninitialize();
        return std::numeric_limits<double>::quiet_NaN();
    }

    IEnumWbemClassObject* pEnumerator = nullptr;
    hres = pSvc->ExecQuery(
        bstr_t("WQL"),
        bstr_t("SELECT CurrentTemperature FROM MSAcpi_ThermalZoneTemperature"),
        WBEM_FLAG_FORWARD_ONLY | WBEM_FLAG_RETURN_IMMEDIATELY,
        nullptr,
        &pEnumerator
    );
    if (FAILED(hres) || !pEnumerator) {
        pSvc->Release();
        pLoc->Release();
        if (uninit) CoUninitialize();
        return std::numeric_limits<double>::quiet_NaN();
    }

    IWbemClassObject* pObj = nullptr;
    ULONG uRet = 0;
    hres = pEnumerator->Next(WBEM_INFINITE, 1, &pObj, &uRet);

    double temp_c = std::numeric_limits<double>::quiet_NaN();
    if (SUCCEEDED(hres) && uRet > 0 && pObj) {
        VARIANT vtProp;
        VariantInit(&vtProp);
        if (SUCCEEDED(pObj->Get(L"CurrentTemperature", 0, &vtProp, nullptr, nullptr))) {
            if (vtProp.vt == VT_I4) temp_c = (vtProp.intVal / 10.0) - 273.15;
            else if (vtProp.vt == VT_UI4) temp_c = (vtProp.ulVal / 10.0) - 273.15;
            else if (vtProp.vt == VT_I2) temp_c = (vtProp.iVal / 10.0) - 273.15;
            else if (vtProp.vt == VT_UI2) temp_c = (vtProp.uiVal / 10.0) - 273.15;
        }
        VariantClear(&vtProp);
        pObj->Release();
    }

    pEnumerator->Release();
    pSvc->Release();
    pLoc->Release();
    if (uninit) CoUninitialize();
    return temp_c;
}

void print_openssl_errors(const std::string& where) {
    unsigned long err = 0;
    std::cerr << where << " failed:\n";
    while ((err = ERR_get_error()) != 0) {
        char buf[256];
        ERR_error_string_n(err, buf, sizeof(buf));
        std::cerr << "  - " << buf << "\n";
    }
}

EVP_PKEY* generate_rsa_key(int bits, int primes) {
    // NOTE:
    // In this toolchain, EVP_PKEY_CTX_set_rsa_keygen_primes() is known to crash
    // for multi-prime RSA, so we use RSA_generate_multi_prime_key() instead.
    if (g_print_key_diagnostics) {
        std::cerr << "[KeyGen] RSA: bits=" << bits << ", primes=" << primes << std::endl;
    }

    BIGNUM* e = BN_new();
    if (!e || BN_set_word(e, RSA_F4) != 1) {
        BN_free(e);
        return nullptr;
    }

    RSA* rsa = RSA_new();
    if (!rsa) {
        BN_free(e);
        return nullptr;
    }

    int rc = 0;
    if (primes <= 2) {
        rc = RSA_generate_key_ex(rsa, bits, e, nullptr);
    } else {
        // primes is total primes (p,q,r,...) in this codebase.
        rc = RSA_generate_multi_prime_key(rsa, bits, primes, e, nullptr);
    }

    if (rc != 1) {
        print_openssl_errors(primes <= 2 ? "RSA_generate_key_ex" : "RSA_generate_multi_prime_key");
        RSA_free(rsa);
        BN_free(e);
        return nullptr;
    }

    EVP_PKEY* pkey = EVP_PKEY_new();
    if (!pkey) {
        RSA_free(rsa);
        BN_free(e);
        return nullptr;
    }

    if (EVP_PKEY_assign_RSA(pkey, rsa) != 1) {
        EVP_PKEY_free(pkey);
        RSA_free(rsa);
        BN_free(e);
        return nullptr;
    }

    BN_free(e);
    return pkey;
}

bool verify_key_crt_details(EVP_PKEY* pkey, int expected_primes, const std::string& label) {
    RSA* rsa = EVP_PKEY_get1_RSA(pkey);
    if (!rsa) {
        std::cerr << "Unable to get RSA from " << label << " key.\n";
        return false;
    }

    const BIGNUM* n = nullptr;
    RSA_get0_key(rsa, &n, nullptr, nullptr);
    int bits = n ? BN_num_bits(n) : 0;
    int extra_count = RSA_get_multi_prime_extra_count(rsa);
    int actual_primes = 2 + ((extra_count > 0) ? extra_count : 0);

    std::cout << "[Key Check] " << label << ": bits=" << bits
              << ", actual_primes=" << actual_primes
              << ", extra_primes=" << extra_count << "\n";

    bool ok = (actual_primes == expected_primes);
    if (!ok) {
        std::cerr << "Prime count mismatch for " << label << ". Expected "
                  << expected_primes << " but got " << actual_primes << ".\n";
    }

    RSA_free(rsa);
    return ok;
}

std::vector<unsigned char> rsa_encrypt(EVP_PKEY* pkey, const std::vector<unsigned char>& plaintext) {
    EVP_PKEY_CTX* ctx = EVP_PKEY_CTX_new(pkey, nullptr);
    if (!ctx) return {};
    if (EVP_PKEY_encrypt_init(ctx) <= 0) {
        EVP_PKEY_CTX_free(ctx);
        return {};
    }
    if (EVP_PKEY_CTX_set_rsa_padding(ctx, RSA_PKCS1_OAEP_PADDING) <= 0) {
        EVP_PKEY_CTX_free(ctx);
        return {};
    }

    size_t out_len = 0;
    if (EVP_PKEY_encrypt(ctx, nullptr, &out_len, plaintext.data(), plaintext.size()) <= 0) {
        EVP_PKEY_CTX_free(ctx);
        return {};
    }

    std::vector<unsigned char> out(out_len);
    if (EVP_PKEY_encrypt(ctx, out.data(), &out_len, plaintext.data(), plaintext.size()) <= 0) {
        EVP_PKEY_CTX_free(ctx);
        return {};
    }
    out.resize(out_len);
    EVP_PKEY_CTX_free(ctx);
    return out;
}

bool init_decrypt_ctx(KeyContext& ctx) {
    ctx.decrypt_ctx = EVP_PKEY_CTX_new(ctx.key, nullptr);
    if (!ctx.decrypt_ctx) return false;
    if (EVP_PKEY_decrypt_init(ctx.decrypt_ctx) <= 0) return false;
    if (EVP_PKEY_CTX_set_rsa_padding(ctx.decrypt_ctx, RSA_PKCS1_OAEP_PADDING) <= 0) return false;
    return true;
}

std::vector<unsigned char> rsa_decrypt_fast(EVP_PKEY_CTX* ctx, const std::vector<unsigned char>& ciphertext) {
    size_t out_len = 0;
    if (EVP_PKEY_decrypt(ctx, nullptr, &out_len, ciphertext.data(), ciphertext.size()) <= 0) return {};
    std::vector<unsigned char> out(out_len);
    if (EVP_PKEY_decrypt(ctx, out.data(), &out_len, ciphertext.data(), ciphertext.size()) <= 0) return {};
    out.resize(out_len);
    return out;
}

bool build_key_context(KeyContext& ctx, int bits, int primes, const std::vector<unsigned char>& plaintext) {
    ctx.key = generate_rsa_key(bits, primes);
    if (!ctx.key) return false;
    if (g_print_key_diagnostics) {
        std::cerr << "[BuildKey] " << ctx.scenario_name << ": key generated (bits=" << bits << ", primes=" << primes << ")" << std::endl;
    }
    ctx.ciphertext = rsa_encrypt(ctx.key, plaintext);
    if (ctx.ciphertext.empty()) {
        EVP_PKEY_free(ctx.key);
        ctx.key = nullptr;
        return false;
    }
    if (g_print_key_diagnostics) {
        std::cerr << "[BuildKey] " << ctx.scenario_name << ": ciphertext encrypted, bytes=" << ctx.ciphertext.size() << std::endl;
    }
    if (!init_decrypt_ctx(ctx)) {
        EVP_PKEY_free(ctx.key);
        ctx.key = nullptr;
        return false;
    }
    if (g_print_key_diagnostics) {
        std::cerr << "[BuildKey] " << ctx.scenario_name << ": decrypt ctx initialized" << std::endl;
    }
    return true;
}

CryptoMode apply_hysteresis(CryptoMode current,
                            const TelemetryInfo& t,
                            const HysteresisThresholds& th,
                            double now_s,
                            std::optional<double>& standard_recovery_start_s) {
    if (current == CryptoMode::Standard) {
        standard_recovery_start_s.reset();
        bool low_battery = (t.battery_pct >= 0 && t.battery_pct < th.switch_to_multiprime_battery_pct);
        bool hot_cpu = (!std::isnan(t.cpu_temp_c) && t.cpu_temp_c > th.switch_to_multiprime_temp_c);
        if (low_battery || hot_cpu) return CryptoMode::MultiPrime;
        return current;
    }

    bool battery_recovered = (t.battery_pct >= 0 && t.battery_pct > th.switch_to_standard_battery_pct);
    bool temperature_recovered = (!std::isnan(t.cpu_temp_c) && t.cpu_temp_c < th.switch_to_standard_temp_c);
    if (battery_recovered && temperature_recovered) {
        if (!standard_recovery_start_s.has_value()) {
            standard_recovery_start_s = now_s;
            return current;
        }
        if ((now_s - *standard_recovery_start_s) >= th.switch_to_standard_hold_s) return CryptoMode::Standard;
        return current;
    }
    standard_recovery_start_s.reset();
    return current;
}

const char* mode_to_name(CryptoMode mode) {
    return mode == CryptoMode::Standard ? "standard_mode" : "efficiency_mode";
}

void write_csv_row(std::ofstream& csv,
                   int iteration,
                   double elapsed_s,
                   const std::string& scenario,
                   int key_size,
                   long long decryption_us,
                   double decryption_s,
                   double joules,
                   int switch_event,
                   const TelemetryInfo& t) {
    csv << iteration << ","
        << std::fixed << std::setprecision(6) << elapsed_s << ","
        << scenario << ","
        << key_size << ","
        << decryption_us << ","
        << std::setprecision(9) << decryption_s << ","
        << std::setprecision(9) << joules << ","
        << switch_event << ","
        << t.battery_pct << ",";
    if (std::isnan(t.cpu_temp_c)) csv << "NaN";
    else csv << std::fixed << std::setprecision(2) << t.cpu_temp_c;
    csv << "," << t.is_charging << "\n";
}

void cleanup_key_context(KeyContext& ctx) {
    EVP_PKEY_CTX_free(ctx.decrypt_ctx);
    EVP_PKEY_free(ctx.key);
    ctx.decrypt_ctx = nullptr;
    ctx.key = nullptr;
}

bool run_adaptive_benchmark(int bits, const RuntimeOptions& options, std::ofstream& csv) {
    std::string msg = "RSA benchmark payload";
    std::vector<unsigned char> plain(msg.begin(), msg.end());

    KeyContext standard_ctx{"standard_mode"};
    KeyContext multiprime_ctx{"efficiency_mode"};

    if (!build_key_context(standard_ctx, bits, 2, plain)) return false;
    if (!build_key_context(multiprime_ctx, bits, 3, plain)) {
        cleanup_key_context(standard_ctx);
        return false;
    }

    bool standard_ok = verify_key_crt_details(standard_ctx.key, 2, "Standard");
    bool multiprime_ok = verify_key_crt_details(multiprime_ctx.key, 3, "Multi-prime");
    if (!standard_ok || !multiprime_ok) {
        cleanup_key_context(standard_ctx);
        cleanup_key_context(multiprime_ctx);
        return false;
    }

    CryptoMode current_mode = CryptoMode::Standard;
    auto bench_t0 = std::chrono::steady_clock::now();
    std::optional<double> standard_recovery_start_s;

    for (int i = 0; i < options.iterations; ++i) {
        BatteryInfo b = get_battery_info();
        double temp = get_system_temperature_c();
        TelemetryInfo t{b.battery_pct, temp, b.is_charging};

        if (options.mock_battery_pct.has_value()) t.battery_pct = *options.mock_battery_pct;
        if (options.mock_temp_c.has_value()) t.cpu_temp_c = *options.mock_temp_c;
        if (options.mock_charging.has_value()) t.is_charging = *options.mock_charging;

        double now_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - bench_t0).count();
        CryptoMode next_mode = apply_hysteresis(current_mode, t, options.thresholds, now_s, standard_recovery_start_s);
        int switch_event = (next_mode != current_mode) ? 1 : 0;
        current_mode = next_mode;

        KeyContext& active_ctx = (current_mode == CryptoMode::Standard) ? standard_ctx : multiprime_ctx;

        auto t1 = std::chrono::high_resolution_clock::now();
        auto dec = rsa_decrypt_fast(active_ctx.decrypt_ctx, active_ctx.ciphertext);
        auto t2 = std::chrono::high_resolution_clock::now();

        if (dec != plain) {
            cleanup_key_context(standard_ctx);
            cleanup_key_context(multiprime_ctx);
            return false;
        }

        auto elapsed = std::chrono::duration<double>(t2 - t1).count();
        long long us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        double joules = elapsed * options.power_watts;
        double since_start = std::chrono::duration<double>(std::chrono::steady_clock::now() - bench_t0).count();

        write_csv_row(csv,
                      i + 1,
                      since_start,
                      mode_to_name(current_mode),
                      bits,
                      us,
                      elapsed,
                      joules,
                      switch_event,
                      t);
    }

    cleanup_key_context(standard_ctx);
    cleanup_key_context(multiprime_ctx);
    return true;
}

double mean_us(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / (double)v.size();
}

double stddev_us(const std::vector<double>& v, double mean) {
    if (v.size() < 2) return 0.0;
    double acc = 0.0;
    for (double x : v) {
        double d = x - mean;
        acc += d * d;
    }
    return std::sqrt(acc / (double)(v.size() - 1));
}

bool run_microbenchmark(int bits, const RuntimeOptions& options) {
    std::string msg = "RSA benchmark payload";
    std::vector<unsigned char> plain(msg.begin(), msg.end());
    KeyContext standard_ctx{"micro_standard"};
    KeyContext multiprime_ctx{"micro_multiprime"};
    if (!build_key_context(standard_ctx, bits, 2, plain)) return false;
    if (!build_key_context(multiprime_ctx, bits, 3, plain)) {
        cleanup_key_context(standard_ctx);
        return false;
    }
    bool standard_ok = verify_key_crt_details(standard_ctx.key, 2, "Micro Standard");
    bool multiprime_ok = verify_key_crt_details(multiprime_ctx.key, 3, "Micro Multi-prime");
    if (!standard_ok || !multiprime_ok) {
        cleanup_key_context(standard_ctx);
        cleanup_key_context(multiprime_ctx);
        return false;
    }

    auto warm = [&](KeyContext& ctx) -> bool {
        for (int i = 0; i < options.warmup_iterations; ++i) {
            auto dec = rsa_decrypt_fast(ctx.decrypt_ctx, ctx.ciphertext);
            if (dec != plain) return false;
        }
        return true;
    };
    if (!warm(standard_ctx) || !warm(multiprime_ctx)) {
        cleanup_key_context(standard_ctx);
        cleanup_key_context(multiprime_ctx);
        return false;
    }

    auto run_mode = [&](KeyContext& ctx) -> std::vector<double> {
        std::vector<double> samples;
        samples.reserve(options.micro_iterations);
        for (int i = 0; i < options.micro_iterations; ++i) {
            auto t1 = std::chrono::high_resolution_clock::now();
            auto dec = rsa_decrypt_fast(ctx.decrypt_ctx, ctx.ciphertext);
            auto t2 = std::chrono::high_resolution_clock::now();
            if (dec != plain) return {};
            double us = std::chrono::duration<double, std::micro>(t2 - t1).count();
            samples.push_back(us);
        }
        return samples;
    };

    std::vector<double> s = run_mode(standard_ctx);
    std::vector<double> m = run_mode(multiprime_ctx);
    if (s.empty() || m.empty()) {
        cleanup_key_context(standard_ctx);
        cleanup_key_context(multiprime_ctx);
        return false;
    }

    double s_mean = mean_us(s);
    double m_mean = mean_us(m);
    double s_sd = stddev_us(s, s_mean);
    double m_sd = stddev_us(m, m_mean);
    double speedup = (m_mean > 0.0) ? (s_mean / m_mean) : 0.0;

    std::ofstream csv(options.micro_out_csv, std::ios::trunc);
    if (csv.is_open()) {
        csv << "mode,iteration,decryption_us\n";
        for (int i = 0; i < (int)s.size(); ++i) csv << "standard," << (i + 1) << "," << s[i] << "\n";
        for (int i = 0; i < (int)m.size(); ++i) csv << "multi_prime," << (i + 1) << "," << m[i] << "\n";
    }

    std::cout << "[Microbenchmark] iterations=" << options.micro_iterations
              << ", warmup=" << options.warmup_iterations << "\n";
    std::cout << "  Standard mean=" << s_mean << " us, sd=" << s_sd << " us\n";
    std::cout << "  Multi-prime mean=" << m_mean << " us, sd=" << m_sd << " us\n";
    std::cout << "  Speedup (Standard/Multi-prime)=" << speedup << "x\n";
    std::cout << "  Saved: " << options.micro_out_csv << "\n";

    cleanup_key_context(standard_ctx);
    cleanup_key_context(multiprime_ctx);
    return true;
}

void print_usage(const char* exe_name) {
    std::cout
        << "Usage:\n"
        << "  " << exe_name << " [out_csv] [--iterations N] [--power-watts W] [--key-bits B]\n"
        << "               [--switch-to-mp-battery PCT] [--switch-to-mp-temp C]\n"
        << "               [--switch-to-std-battery PCT] [--switch-to-std-temp C]\n"
        << "               [--switch-to-std-hold-seconds S]\n"
        << "               [--mock-battery PCT] [--mock-temp C] [--mock-charging 0|1]\n"
        << "               [--run-microbenchmark 0|1] [--micro-iterations N] [--warmup-iterations N]\n"
        << "               [--micro-out-csv PATH]\n\n"
        << "Defaults:\n"
        << "  switch_to_multiprime: battery < 30 OR temp > 65\n"
        << "  switch_to_standard:   battery > 40 AND temp < 55 sustained for 120s\n";
}

bool parse_int(const std::string& s, int& out) {
    try {
        size_t pos = 0;
        int v = std::stoi(s, &pos);
        if (pos != s.size()) return false;
        out = v;
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_double(const std::string& s, double& out) {
    try {
        size_t pos = 0;
        double v = std::stod(s, &pos);
        if (pos != s.size()) return false;
        out = v;
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_args(int argc, char* argv[], RuntimeOptions& options) {
    int i = 1;
    if (i < argc && std::string(argv[i]).rfind("--", 0) != 0) {
        options.out_csv = argv[i];
        ++i;
    }

    while (i < argc) {
        std::string arg = argv[i];
        auto need_value = [&](const std::string& flag) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << flag << "\n";
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return false;
        } else if (arg == "--iterations") {
            const char* v = need_value(arg);
            int parsed = 0;
            if (!v || !parse_int(v, parsed) || parsed <= 0) return false;
            options.iterations = parsed;
        } else if (arg == "--power-watts") {
            const char* v = need_value(arg);
            double parsed = 0.0;
            if (!v || !parse_double(v, parsed) || parsed <= 0.0) return false;
            options.power_watts = parsed;
        } else if (arg == "--key-bits") {
            const char* v = need_value(arg);
            int parsed = 0;
            if (!v || !parse_int(v, parsed) || parsed < 1024) return false;
            options.key_bits = parsed;
        } else if (arg == "--switch-to-mp-battery") {
            const char* v = need_value(arg);
            int parsed = 0;
            if (!v || !parse_int(v, parsed)) return false;
            options.thresholds.switch_to_multiprime_battery_pct = parsed;
        } else if (arg == "--switch-to-mp-temp") {
            const char* v = need_value(arg);
            double parsed = 0.0;
            if (!v || !parse_double(v, parsed)) return false;
            options.thresholds.switch_to_multiprime_temp_c = parsed;
        } else if (arg == "--switch-to-std-battery") {
            const char* v = need_value(arg);
            int parsed = 0;
            if (!v || !parse_int(v, parsed)) return false;
            options.thresholds.switch_to_standard_battery_pct = parsed;
        } else if (arg == "--switch-to-std-temp") {
            const char* v = need_value(arg);
            double parsed = 0.0;
            if (!v || !parse_double(v, parsed)) return false;
            options.thresholds.switch_to_standard_temp_c = parsed;
        } else if (arg == "--switch-to-std-hold-seconds") {
            const char* v = need_value(arg);
            double parsed = 0.0;
            if (!v || !parse_double(v, parsed) || parsed < 0.0) return false;
            options.thresholds.switch_to_standard_hold_s = parsed;
        } else if (arg == "--mock-battery") {
            const char* v = need_value(arg);
            int parsed = 0;
            if (!v || !parse_int(v, parsed)) return false;
            options.mock_battery_pct = parsed;
        } else if (arg == "--mock-temp") {
            const char* v = need_value(arg);
            double parsed = 0.0;
            if (!v || !parse_double(v, parsed)) return false;
            options.mock_temp_c = parsed;
        } else if (arg == "--mock-charging") {
            const char* v = need_value(arg);
            int parsed = 0;
            if (!v || !parse_int(v, parsed) || (parsed != 0 && parsed != 1)) return false;
            options.mock_charging = parsed;
        } else if (arg == "--run-microbenchmark") {
            const char* v = need_value(arg);
            int parsed = 0;
            if (!v || !parse_int(v, parsed) || (parsed != 0 && parsed != 1)) return false;
            options.run_microbenchmark = (parsed == 1);
        } else if (arg == "--micro-iterations") {
            const char* v = need_value(arg);
            int parsed = 0;
            if (!v || !parse_int(v, parsed) || parsed <= 0) return false;
            options.micro_iterations = parsed;
        } else if (arg == "--warmup-iterations") {
            const char* v = need_value(arg);
            int parsed = 0;
            if (!v || !parse_int(v, parsed) || parsed < 0) return false;
            options.warmup_iterations = parsed;
        } else if (arg == "--micro-out-csv") {
            const char* v = need_value(arg);
            if (!v) return false;
            options.micro_out_csv = v;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return false;
        }
        ++i;
    }
    return true;
}

int main(int argc, char* argv[]) {
    RuntimeOptions options;
    if (!parse_args(argc, argv, options)) {
        print_usage(argv[0]);
        return 1;
    }
    g_print_key_diagnostics = options.print_key_diagnostics;

    std::ofstream csv(options.out_csv, std::ios::trunc);
    if (!csv.is_open()) return 1;
    csv << "iteration,elapsed_s,scenario,key_size,decryption_us,decryption_s,energy_j,switch_event,battery_pct,cpu_temp_c,is_charging\n";

    if (!run_adaptive_benchmark(options.key_bits, options, csv)) return 1;
    if (options.run_microbenchmark && !run_microbenchmark(options.key_bits, options)) return 1;

    std::cout << "Done: " << options.out_csv << "\n";
    return 0;
}
