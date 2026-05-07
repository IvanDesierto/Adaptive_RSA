# Adaptive RSA Handshake Signal Design

## Goal

Use a 1-byte control signal to tell peers which RSA public key to use (`Standard` 2-prime or `Multi-prime` 3-prime), without introducing a separate control channel.

## Signal Encoding (1 byte)

- `0x00`: no mode change (keep current key)
- `0x01`: switch to standard RSA public key
- `0x02`: switch to multi-prime RSA public key
- `0x80` bit reserved for ACK (optional)

## Transport Placement

### TLS piggyback option

Attach the 1-byte signal as a private extension in an existing TLS handshake message (for example, `EncryptedExtensions` in TLS 1.3). The extension is integrity protected by TLS.

### UDP piggyback option

Insert the 1-byte signal in an existing application header byte (or append to a fixed metadata header) in each datagram that already carries encrypted payload.

## Protocol State Machine (high level)

1. Receiver computes mode from hysteresis logic:
   - switch to multi-prime when `battery < 20 OR temperature > 70C`
   - switch back to standard when `battery > 25 AND temperature < 65C`
2. If mode changes, receiver emits control byte (`0x01` or `0x02`) on next outgoing packet.
3. Sender reads control byte and updates active peer public key for subsequent encryptions.
4. Sender ACKs using high bit (`0x81` or `0x82`) if ACK mode is enabled.
5. Receiver retries control byte until ACK or timeout, then continues with current mode.

## Pseudocode

```text
state current_mode = STANDARD
state peer_mode = STANDARD

on telemetry_update(battery, temp):
    next_mode = current_mode
    if current_mode == STANDARD:
        if battery < 20 or temp > 70:
            next_mode = MULTIPRIME
    else:
        if battery > 25 and temp < 65:
            next_mode = STANDARD

    if next_mode != current_mode:
        current_mode = next_mode
        ctrl = (next_mode == STANDARD) ? 0x01 : 0x02
        queue_control_byte(ctrl)

on incoming_packet(pkt):
    ctrl = parse_control_byte(pkt)
    if ctrl == 0x01: peer_mode = STANDARD
    if ctrl == 0x02: peer_mode = MULTIPRIME
    if ack_enabled and (ctrl == 0x01 or ctrl == 0x02):
        send_control_byte(ctrl | 0x80)

on encrypt_outgoing(payload):
    pubkey = (peer_mode == STANDARD) ? peer_std_key : peer_mp_key
    return rsa_encrypt(pubkey, payload)
```

## Safety Notes

- Keep both key pairs valid during transition window (grace period).
- Authenticate control byte via existing channel security (TLS transcript MAC or datagram-level AEAD).
- Ignore unknown control values for forward compatibility.
