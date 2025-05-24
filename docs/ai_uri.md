# AI URI Specification

This document describes the formal structure of the `ai://` Uniform Resource Identifier (URI) used by Avalan to reference models hosted either locally or on remote vendors.  The grammar below follows a Backus--Naur Form (BNF) style notation.

```
<ai-uri>        ::= ["ai://"] <authority> ["/" <path>] ["?" <query>]
<authority>     ::= [<userinfo> "@"] <hostport>
<userinfo>      ::= <user> [":" <password>]
<hostport>      ::= <host> [":" <port>]
<host>          ::= <vendor> | <model-host>
<vendor>        ::= "anthropic" | "deepseek" | "google" | \
                    "groq" | "huggingface" | "local" | \
                    "openai" | "openrouter" | "ollama"
<path>          ::= 1*( pchar | "/" )
<query>         ::= *( pchar | "=" | "&" )
```

- The URI **scheme** must be `ai`.  If omitted, the parser assumes the prefix `ai://`.
- `<userinfo>` is optional and typically holds an API key used when accessing a remote vendor.  The password component may be omitted or empty.
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
| `user`    | Username from `<userinfo>` (e.g. an API key), if present.     |
| `password`| Password from `<userinfo>` if provided and non-empty.         |
| `model_id`| Normalised model identifier derived from `<path>`.            |
| `params`  | Mapping of query parameters.                                  |

When `vendor` is `None`, the model is considered local.  Otherwise, the URI describes a remote engine hosted by the given vendor.

## Examples

| URI                                        | Parsed Result                                                   |
|--------------------------------------------|----------------------------------------------------------------|
| `tiiuae/Falcon-E-3B-Instruct`               | local model `tiiuae/Falcon-E-3B-Instruct`                      |
| `ai://local/tiiuae/Falcon-E-3B-Instruct`    | same as above                                                  |
| `ai://messi_api_key@openai/gpt-4o`         | vendor `openai`, user `messi_api_key`, model `gpt-4o`          |
| `ai://hf_key:@huggingface/meta-llama/Llama-3-8B-Instruct` | vendor `huggingface`, user `hf_key`, model `meta-llama/Llama-3-8B-Instruct` |
| `ai://ollama/llama3`                        | vendor `ollama`, model `llama3`                                |

These examples correspond to those used in the test-suite and highlight both local and remote forms.

