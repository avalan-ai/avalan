"""Enforce task-input security policy at host adapter boundaries."""

from .entities import (
    InputQuestion,
    InputRequest,
    MultipleSelectionQuestion,
    SingleSelectionQuestion,
)
from .error import InputErrorCode, InputValidationError

from re import compile as compile_pattern
from unicodedata import category, normalize

_CREDENTIAL_SEPARATOR = r"[\s._-]*"
_PASSWORD_SKELETON = (
    rf"(?<![a-z0-9])p{_CREDENTIAL_SEPARATOR}a"
    rf"{_CREDENTIAL_SEPARATOR}s{_CREDENTIAL_SEPARATOR}s"
    rf"{_CREDENTIAL_SEPARATOR}w{_CREDENTIAL_SEPARATOR}o"
    rf"{_CREDENTIAL_SEPARATOR}r{_CREDENTIAL_SEPARATOR}d"
    rf"(?:{_CREDENTIAL_SEPARATOR}s)?(?![a-z0-9])"
)
_API_KEY_SKELETON = (
    rf"(?<![a-z0-9])a{_CREDENTIAL_SEPARATOR}p"
    rf"{_CREDENTIAL_SEPARATOR}i{_CREDENTIAL_SEPARATOR}k"
    rf"{_CREDENTIAL_SEPARATOR}e{_CREDENTIAL_SEPARATOR}y"
    rf"(?:{_CREDENTIAL_SEPARATOR}s)?(?![a-z0-9])"
)
_CREDENTIAL_SKELETON_TRANSLATION = str.maketrans(
    {
        "@": "a",
        "0": "o",
        "4": "a",
        "\u0430": "a",
        "\u0435": "e",
        "\u0456": "i",
        "\u043a": "k",
        "\u043e": "o",
        "\u0440": "p",
        "\u0441": "c",
        "\u0443": "y",
        "\u0445": "x",
    }
)
_SENSITIVE_REQUEST_PATTERN = compile_pattern(
    rf"(?:{_PASSWORD_SKELETON}|"
    r"\b(?:ssh[\s._-]+)?pass[\s._-]*phrases?\b|"
    r"\bpass[\s._-]*codes?\b|"
    r"\bsecrets?(?:[\s._-]*(?:materials?|tokens?|values?))*\b|"
    r"\bcredentials?\b|"
    rf"{_API_KEY_SKELETON}|"
    r"\bapi[\s._-]*tokens?\b|"
    r"\b(?:access|refresh|bearer|session|identity)[\s._-]*tokens?\b|"
    r"\bgithub[\s._-]*tokens?\b|"
    r"\bauth(?:entication|orization|orisation)?[\s._-]*tokens?\b|"
    r"\bprivate[\s._-]*keys?\b|\bpayments?\b|"
    r"\b(?:credit|payment)[\s._-]*cards?\b|"
    r"\bcard[\s._-]*(?:numbers?|security[\s._-]*codes?|"
    r"expir(?:y|ation)[\s._-]*dates?|"
    r"details?|credentials?|materials?)\b|"
    r"\bpayment[\s._-]*(?:details?|credentials?|materials?)\b|"
    r"\b(?:cvvs?|cvcs?|ibans?)\b|"
    r"\bbank[\s._-]*(?:accounts?|credentials?)\b|"
    r"\baccount[\s._-]*(?:numbers?|credentials?|details?|materials?)\b|"
    r"\brouting[\s._-]*(?:numbers?|codes?|details?|materials?)\b|"
    r"\b(?:mfa|2fa|otp|totp|hotp)s?\b|"
    r"\bpin[\s._-]*(?:codes?|numbers?|values?)\b|"
    r"\bone[\s._-]*time[\s._-]*(?:codes?|passwords?)\b|"
    r"\b(?:log[\s._-]*in|sign[\s._-]*in|phone|mobile|sms|"
    r"text[\s._-]*message)[\s._-]*(?:(?:verification|security|"
    r"one[\s._-]*time)[\s._-]*)?codes?\b|"
    r"\b(?:log[\s._-]*in|sign[\s._-]*in)\b[^.!?\n]{0,80}\bcodes?\b|"
    r"\bcodes?\b[^.!?\n]{0,80}\b(?:log[\s._-]*in|sign[\s._-]*in|"
    r"phone|mobile|sms|text[\s._-]*message)\b|"
    r"\b(?:authenticator(?:[\s._-]*app)?|recovery|security|verification)"
    r"[\s._-]*codes?\b|"
    r"\bauth(?:entication|orization|orisation)?\b|"
    r"\bauthentication[\s._-]*challenges?\b|"
    r"\b(?:sign[\s._-]*in[\s._-]*challenge[\s._-]*responses?|"
    r"challenge[\s._-]*responses?[\s._-]*from(?:[\s._-]*your)?"
    r"[\s._-]*sign[\s._-]*in[\s._-]*pages?)\b|\boauth\b)"
)
_PIN_AUTH_QUALIFIER = (
    r"(?:access|account|auth(?:entication|orization|orisation)?|"
    r"credentials?|identity|log[\s._-]*in|security|sign[\s._-]*in|"
    r"unlock|verification)"
)
_PIN_AUTH_CONTEXT_PATTERN = compile_pattern(
    rf"(?:\b{_PIN_AUTH_QUALIFIER}\b[^.!?\n]{{0,80}}\bpins?\b|"
    rf"\bpins?\b[^.!?\n]{{0,80}}\b{_PIN_AUTH_QUALIFIER}\b)"
)
_PIN_TERM_PATTERN = compile_pattern(r"\bpins?\b")
_NON_CREDENTIAL_PIN_CONTEXT_PATTERN = compile_pattern(
    r"\b(?:location|map)[\s._-]+pins?\b|"
    r"\bpins?[\s._-]+(?:location|map)\b|"
    r"\ba[\s._-]+pins?[\s._-]+(?:in|on)[\s._-]+"
    r"(?:the[\s._-]+)?(?:location|map)\b"
)
_POSSESSIVE_PIN_CONTEXT_PATTERN = compile_pattern(
    r"\b(?:my|our|their|your)\b[^.!?;\n]{0,40}\bpins?\b"
)
_COLLECTION_INTENT_PATTERN = compile_pattern(
    r"\b(?:answer|change|choose|complete|confirm|create|disclose|enter|"
    r"give|input|paste|pick|provide|re[\s._-]*enter|re[\s._-]*type|"
    r"repeat|reset|respond|reveal|select|send|set|share|submi(?:t|tted|tting)|"
    r"supply|tell|type|update|use|validate|verify)\b"
)
_POSSESSIVE_QUESTION_PATTERN = compile_pattern(
    r"\b(?:what|which)\s+(?:are|is|was|were)\b[^.!?;\n]{0,80}"
    r"\b(?:my|our|their|your)\b"
)
_NEGATION_PATTERN = compile_pattern(
    r"(?:\b(?:avoid|cannot|can't|do[\s._-]*not|don't|must[\s._-]*not|"
    r"never|no[\s._-]*need[\s._-]*to|should[\s._-]*not|without|"
    r"won't|would[\s._-]*not)\b)"
)
_CLAUSE_BOUNDARY_PATTERN = compile_pattern(r"[.!?;\n]")
_NEGATION_RESET_PATTERN = compile_pattern(
    r"\b(?:but|however|instead|nevertheless)\b"
)
_DISCUSSION_PATTERN = compile_pattern(
    r"\b(?:architecture|describe|design|discuss|document|explain|overview|"
    r"policy|policies|requirements?|teach|unsafe|warn|why)\b"
)
_DISCUSSION_QUALIFIER_PATTERN = compile_pattern(
    r"^[\s._-]*(?:(?:collection|handling|management|rotation|storage)"
    r"[\s._-]*)?(?:architecture|design|documentation|handling|management|"
    r"overview|policy|policies|requirements?|rotation|storage)\b"
)
_AUTH_CONTEXT_PATTERN = compile_pattern(
    r"\b(?:account|auth(?:entication|orization|orisation)?|identity|"
    r"log[\s._-]*in|mfa|oauth|otp|security|sign[\s._-]*in|"
    r"verification)\b"
)
_SPLIT_AUTH_COLLECTION_PATTERN = compile_pattern(
    r"\b(?:answer|confirm|enter|give|input|paste|provide|respond|reveal|"
    r"send|share|submit|supply|tell|type|verify)\b"
    r"[^.!?;\n]{0,40}\b(?:the|this|your)?[\s._-]*(?:codes?|pins?)\b"
    r"[\s._-]*(?:value)?\s*$"
)
_LABEL_DECORATION_PATTERN = compile_pattern(
    r"(?:[\s:;,./_!?*()[\]{}-]|"
    r"\b(?:field|optional|please|required|the|value|your)\b)*"
)
_SECRET_VALUE_PATTERN = compile_pattern(
    r"(?<![A-Za-z0-9])(?:"
    r"sk-(?:proj-)?[A-Za-z0-9_-]{16,}|"
    r"(?:sk|pk)_(?:live|test)_[A-Za-z0-9]{12,}|"
    r"AIza[A-Za-z0-9_-]{20,}|"
    r"(?:AKIA|ASIA)[A-Z0-9]{16}|"
    r"github_pat_[A-Za-z0-9_]{20,}|"
    r"gh[pousr]_[A-Za-z0-9]{20,}|"
    r"xox[baprs]-[A-Za-z0-9-]{10,}|"
    r"api[_-]?key[:=._-][A-Za-z0-9_./+=-]{12,}"
    r")(?![A-Za-z0-9])"
)
_CARD_NUMBER_CANDIDATE_PATTERN = compile_pattern(
    r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"
)


def task_input_requires_sensitive_flow(request: InputRequest) -> bool:
    """Return whether a request must use a purpose-built sensitive flow."""
    assert type(request) is InputRequest
    return task_input_questions_require_sensitive_flow(
        request.questions,
        surrounding_text=(request.reason, request.context_label or ""),
    )


def task_input_questions_require_sensitive_flow(
    questions: tuple[InputQuestion, ...],
    *,
    surrounding_text: tuple[str, ...] = (),
) -> bool:
    """Return whether questions require a purpose-built sensitive flow."""
    values = list(surrounding_text)
    if any(_is_bare_pin_label(value) for value in surrounding_text):
        return True
    for question in questions:
        question_values = (
            str(question.question_id),
            question.prompt,
            question.header or "",
            question.help_text or "",
        )
        values.extend(question_values)
        if any(_is_bare_pin_label(value) for value in question_values[1:]):
            return True
        if _separate_fields_require_authentication(
            question_values,
            surrounding_text,
        ):
            return True
        default_value = getattr(question, "default_value", None)
        if isinstance(default_value, str):
            values.append(default_value)
        elif isinstance(default_value, tuple):
            values.extend(str(value) for value in default_value)
        if isinstance(
            question,
            (SingleSelectionQuestion, MultipleSelectionQuestion),
        ):
            for choice in question.choices:
                choice_presentation = (
                    choice.label,
                    choice.description or "",
                )
                values.extend(
                    (
                        str(choice.value),
                        *choice_presentation,
                    )
                )
                if any(
                    _is_bare_pin_label(value) for value in choice_presentation
                ):
                    return True
    return any(
        _sensitive_text(value) or _looks_like_secret_value(value)
        for value in values
    )


def enforce_task_input_request_policy(
    request: InputRequest,
    path: str = "request",
) -> None:
    """Reject requests that require a purpose-built sensitive flow."""
    if task_input_requires_sensitive_flow(request):
        _raise_sensitive_request(path)


def enforce_task_input_questions_policy(
    questions: tuple[InputQuestion, ...],
    path: str = "questions",
    *,
    surrounding_text: tuple[str, ...] = (),
) -> None:
    """Reject questions that require a purpose-built sensitive flow."""
    if task_input_questions_require_sensitive_flow(
        questions,
        surrounding_text=surrounding_text,
    ):
        _raise_sensitive_request(path)


def _raise_sensitive_request(path: str) -> None:
    assert isinstance(path, str) and path
    raise InputValidationError(
        InputErrorCode.PROHIBITED_INPUT,
        path,
        "sensitive or authentication input requires a separate flow",
    )


def _sensitive_text(value: str) -> bool:
    normalized = _credential_security_text(value)
    return any(
        _sensitive_clause(clause.strip())
        for clause in _CLAUSE_BOUNDARY_PATTERN.split(normalized)
        if clause.strip()
    )


def _sensitive_clause(normalized: str) -> bool:
    sensitive = _SENSITIVE_REQUEST_PATTERN.search(normalized)
    pin_context = _contains_sensitive_pin_context(normalized)
    if sensitive is None and not pin_context:
        return False
    if _has_unnegated_collection_intent(normalized):
        if pin_context:
            return True
        return any(
            not _discussion_qualified(normalized, match.end())
            for match in _SENSITIVE_REQUEST_PATTERN.finditer(normalized)
        )
    if _DISCUSSION_PATTERN.search(normalized) is not None:
        return False
    if (
        _COLLECTION_INTENT_PATTERN.search(normalized) is not None
        or _POSSESSIVE_QUESTION_PATTERN.search(normalized) is not None
    ):
        return False
    if pin_context:
        return True
    assert sensitive is not None
    remainder = normalized[: sensitive.start()] + normalized[sensitive.end() :]
    return _LABEL_DECORATION_PATTERN.fullmatch(remainder) is not None


def _discussion_qualified(value: str, sensitive_end: int) -> bool:
    return (
        _DISCUSSION_QUALIFIER_PATTERN.match(value[sensitive_end:]) is not None
    )


def _contains_sensitive_pin_context(value: str) -> bool:
    pin_spans = tuple(
        match.span() for match in _PIN_TERM_PATTERN.finditer(value)
    )
    if not pin_spans:
        return False
    if any(
        auth_match.start() <= pin_start and pin_end <= auth_match.end()
        for auth_match in _PIN_AUTH_CONTEXT_PATTERN.finditer(value)
        for pin_start, pin_end in pin_spans
    ):
        return True
    noncredential_spans = tuple(
        match.span()
        for match in _NON_CREDENTIAL_PIN_CONTEXT_PATTERN.finditer(value)
    )
    credential_pin_spans = tuple(
        pin_span
        for pin_span in pin_spans
        if not any(
            context_start <= pin_span[0] and pin_span[1] <= context_end
            for context_start, context_end in noncredential_spans
        )
    )
    if not credential_pin_spans:
        return False
    return (
        _has_unnegated_collection_intent(value)
        or _POSSESSIVE_PIN_CONTEXT_PATTERN.search(value) is not None
    )


def _is_bare_pin_label(value: str) -> bool:
    normalized = _credential_security_text(value)
    pin_matches = tuple(_PIN_TERM_PATTERN.finditer(normalized))
    if len(pin_matches) != 1:
        return False
    pin_match = pin_matches[0]
    remainder = normalized[: pin_match.start()] + normalized[pin_match.end() :]
    return _LABEL_DECORATION_PATTERN.fullmatch(remainder) is not None


def _separate_fields_require_authentication(
    question_values: tuple[str, ...],
    surrounding_text: tuple[str, ...],
) -> bool:
    fields = (*surrounding_text, *question_values)
    normalized = tuple(_credential_security_text(value) for value in fields)
    contexts = tuple(
        value
        for value in normalized
        if _AUTH_CONTEXT_PATTERN.search(value) is not None
        and _DISCUSSION_PATTERN.search(value) is None
    )
    if not contexts:
        return False
    return any(
        _SPLIT_AUTH_COLLECTION_PATTERN.search(value) is not None
        for value in normalized
    )


def _has_unnegated_collection_intent(value: str) -> bool:
    matches = (
        *_COLLECTION_INTENT_PATTERN.finditer(value),
        *_POSSESSIVE_QUESTION_PATTERN.finditer(value),
    )
    for match in sorted(matches, key=lambda item: item.start()):
        prefix = value[: match.start()]
        boundaries = tuple(_CLAUSE_BOUNDARY_PATTERN.finditer(prefix))
        if boundaries:
            prefix = prefix[boundaries[-1].end() :]
        resets = tuple(_NEGATION_RESET_PATTERN.finditer(prefix))
        if resets:
            prefix = prefix[resets[-1].end() :]
        if _NEGATION_PATTERN.search(prefix[-80:]) is None:
            return True
    return False


def _normalize_security_text(value: str) -> str:
    normalized = normalize("NFKC", value)
    return "".join(
        ("-" if category(character) == "Pd" else character)
        for character in normalized
        if category(character) != "Cf"
    )


def _credential_security_text(value: str) -> str:
    return (
        _normalize_security_text(value)
        .casefold()
        .translate(_CREDENTIAL_SKELETON_TRANSLATION)
    )


def _looks_like_secret_value(value: str) -> bool:
    normalized = _normalize_security_text(value)
    if _SECRET_VALUE_PATTERN.search(normalized):
        return True
    return any(
        _passes_luhn_check("".join(filter(str.isdigit, candidate.group())))
        for candidate in _CARD_NUMBER_CANDIDATE_PATTERN.finditer(normalized)
    )


def _passes_luhn_check(value: str) -> bool:
    digits = [int(character) for character in value]
    checksum = sum(digits[-1::-2]) + sum(
        sum(divmod(2 * digit, 10)) for digit in digits[-2::-2]
    )
    return checksum % 10 == 0
