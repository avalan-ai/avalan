# AI URI Specification

This document describes the formal structure of the `ai://` Uniform Resource Identifier (URI) used by Avalan to reference models hosted either locally or on remote vendors.  The grammar below follows a Backus--Naur Form (BNF) style notation.

```
<ai-uri>        ::= ["ai://"] <authority> ["/" <path>] ["?" <query>]
<authority>     ::= [<userinfo> "@"] <hostport>
<userinfo>      ::= <user> [":" <password>]
<hostport>      ::= <host> [":" <port>]
<host>          ::= <vendor> | <model-host>
<vendor>        ::= "anthropic" | "anyscale" | "bedrock" | "deepinfra" | \
                    "deepseek" | "google" | "groq" | "huggingface" | \
                    "hyperbolic" | "local" | "openai" | "openrouter" | \
                    "ollama" | "litellm" | "together"
<path>          ::= 1*( pchar | "/" )
<query>         ::= *( pchar | "=" | "&" )
```

- The URI **scheme** must be `ai`.  If omitted, the parser assumes the prefix `ai://`.
- `<userinfo>` is optional. When only the username is present it is interpreted as the vendor access key. If a password is also supplied, the username denotes the token lookup mechanism and the password the key used to retrieve the vendor access token. Use `secret` to read from the secret backend or `env` to read from an environment variable.
- `<host>` identifies either a known vendor (from the list above) or a local model host.  When the host is not a recognised vendor or equals `local`, the URI is interpreted as a local model reference.
- `<path>` denotes the model identifier.  For local models the first segment of the path can contain a host component that becomes part of the model identifier.  For remote vendors the path is used directly as the model identifier.
- `<query>` parameters are parsed into key–value pairs and attached to the resulting `EngineUri` object.

## Semantics

Parsing an AI URI results in an `EngineUri` structure containing:

| Field     | Description                                                   |
|-----------|---------------------------------------------------------------|
| `vendor`  | Vendor name if `<host>` matches a recognised vendor, otherwise `None`. |
| `host`    | `<host>` when it is a vendor, else `None`.                    |
| `port`    | Optional port when supplied for a vendor.                     |
| `user`    | Value of `<user>` when the password is omitted, otherwise the token lookup mechanism (`secret` or `env`). |
| `password`| Value of `<password>` when supplied; used as the lookup key or environment variable name. |
| `model_id`| Normalised model identifier derived from `<path>`.            |
| `params`  | Mapping of query parameters.                                  |

When `vendor` is `None`, the model is considered local.  Otherwise, the URI describes a remote engine hosted by the given vendor.

## Backend-specific parameters

Local engines can select a backend with `backend=<name>`. For DS4, use
`backend=ds4` with a DS4-supported DeepSeek V4 Flash GGUF file:

```bash
ai://local/./ds4flash.gguf?backend=ds4&ds4_ctx=4096&ds4_native_backend=metal
```

Relative local paths are written as URI paths. Absolute paths use a double
slash after `local`, or an encoded leading slash:

```bash
ai://local/../pyds4/.local/ds4/ds4flash.gguf?backend=ds4
ai://local//Users/me/models/ds4flash.gguf?backend=ds4
ai://local/%2FUsers/me/DS4%20models/ds4flash.gguf?backend=ds4
```

DS4-specific parameters must use the `ds4_` prefix. Common keys are
`ds4_ctx`, `ds4_native_backend`, `ds4_mtp`, `ds4_mtp_draft`,
`ds4_mtp_margin`, `ds4_warm_weights`, and `ds4_quality`. Unknown `ds4_`
parameters are rejected. The DS4 backend is not a generic GGUF loader, does
not yet support native tool calls, and treats CPU mode as a debug/reference
path only.

## Examples

| URI                                        | Parsed Result                                                   |
|--------------------------------------------|----------------------------------------------------------------|
| `tiiuae/Falcon-E-3B-Instruct`               | local model `tiiuae/Falcon-E-3B-Instruct`                      |
| `ai://local/tiiuae/Falcon-E-3B-Instruct`    | same as above                                                  |
| `ai://messi_api_key@openai/gpt-4o`         | vendor `openai`, user `messi_api_key`, model `gpt-4o`          |
| `ai://hf_key:@huggingface/meta-llama/Llama-3-8B-Instruct` | vendor `huggingface`, user `hf_key`, model `meta-llama/Llama-3-8B-Instruct` |
| `ai://secret:openai_key@openai/gpt-4o`     | vendor `openai`, secret key `openai_key`, model `gpt-4o`       |
| `ai://env:ANTHROPIC_API_KEY@anthropic/claude-sonnet-4-6` | vendor `anthropic`, env var `ANTHROPIC_API_KEY`, model `claude-sonnet-4-6` |
| `ai://ollama/llama3`                        | vendor `ollama`, model `llama3`                                |
| `ai://litellm/gpt-3.5-turbo`                | vendor `litellm`, model `gpt-3.5-turbo`                        |
| `ai://tg_key@together/mistral-7b`           | vendor `together`, model `mistral-7b`                          |
| `ai://local/./ds4flash.gguf?backend=ds4&ds4_ctx=4096` | local DS4 GGUF with DS4 backend config             |

These examples correspond to those used in the test-suite and highlight both local and remote forms.
