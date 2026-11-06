"""The navigation allowlist (``configs/routes.yaml``).

The router can return only the semantic route IDs listed in this file. They
come from the navigation registry in ``knowledge/facts.md``. No page paths or
URLs have been chosen for the website, so the file holds route IDs and the
phrases that name them, never a path or URL: unknown keys such as ``url`` are
rejected at every level.

* ``routes``: each route ID with the aliases that name it;
* ``ambiguous_terms`` (optional): phrases that name two or more routes, such
  as "research". They never select a route; the router reports them as
  ambiguous.

Phrases are compared by their key (``mascot_lab.phrases.phrase_key``), so
"Q-MIND", "the QMIND page", and "QMIND" are one alias. Two phrases with the
same key collide, and the file is rejected. Loading reuses the strict YAML
loader and validators of ``mascot_lab.config`` and fails closed with
``ConfigError``.
"""

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import yaml

from mascot_lab._textfile import read_text_file, sha256_hex
from mascot_lab.config import _StrictLoader, _check_keys, _mapping, _text
from mascot_lab.errors import ConfigError
from mascot_lab.paths import PROJECT_ROOT, display_path
from mascot_lab.phrases import CONNECTORS, GENERIC_WORDS, core_words, normalize_text, tokenize

ROUTES_CONFIG_PATH = PROJECT_ROOT / "configs" / "routes.yaml"
SCHEMA_VERSION = 1

_ROOT_KEYS = frozenset({"schema_version", "routes", "ambiguous_terms"})
_ROOT_REQUIRED_KEYS = frozenset({"schema_version", "routes"})
_ROUTE_KEYS = frozenset({"id", "aliases"})
_TERM_KEYS = frozenset({"term", "routes"})

# A semantic route ID such as "project_qmind".
_ROUTE_ID = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*")
# A plain name: letters and digits separated by spaces, periods, hyphens, or apostrophes.
_PHRASE = re.compile(r"[^\W_]+(?:[ .'’-]+[^\W_]+)*")
# A period between letters or digits, as in a domain name.
_DOTTED = re.compile(r"[^\W_]\.[^\W_]")
_BARE_DOMAIN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*(?:\.[a-z0-9]+(?:-[a-z0-9]+)*)+")


@dataclass(frozen=True, slots=True)
class Route:
    """An allowlisted destination: a semantic route ID and the aliases that name it."""

    id: str
    aliases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AmbiguousTerm:
    """A phrase that names several routes, so it never selects one."""

    term: str
    route_ids: tuple[str, ...]  # in configuration order


@dataclass(frozen=True, slots=True)
class Destination:
    """What a phrase key stands for: the phrase as configured and its route IDs.

    An alias has one route ID; an ambiguous term has two or more.
    """

    phrase: str
    route_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RoutesConfig:
    """The validated navigation allowlist."""

    schema_version: int
    routes: tuple[Route, ...]
    ambiguous_terms: tuple[AmbiguousTerm, ...]
    destinations: Mapping[str, Destination]  # by phrase key; one entry per configured phrase
    domain_names: tuple[str, ...]  # normalized phrases that are bare domain names
    source: str
    sha256: str  # of the canonical JSON form, so comments and layout do not affect it

    @property
    def route_ids(self) -> tuple[str, ...]:
        return tuple(route.id for route in self.routes)


def load_routes_config(path: str | os.PathLike[str] | None = None) -> RoutesConfig:
    """Load and validate ``path`` (default: ``configs/routes.yaml``)."""
    file_path = ROUTES_CONFIG_PATH if path is None else Path(path)
    text = read_text_file(file_path, error=ConfigError, label="route configuration")
    return parse_routes_config(text, source=display_path(file_path))


def parse_routes_config(text: str, *, source: str = "<string>") -> RoutesConfig:
    """Validate allowlist ``text``; raise ``ConfigError`` on any problem."""
    try:
        data = yaml.load(text, Loader=_StrictLoader)  # _StrictLoader subclasses SafeLoader
    except yaml.YAMLError as exc:
        raise ConfigError(f"{source}: invalid YAML: {exc}") from exc

    root = _mapping(data, f"{source}: configuration root")
    _check_keys(root, f"{source}: configuration root", _ROOT_KEYS, _ROOT_REQUIRED_KEYS)
    version = root["schema_version"]
    if type(version) is not int or version != SCHEMA_VERSION:
        raise ConfigError(f"{source}: schema_version must be {SCHEMA_VERSION}, found {version!r}")

    destinations: dict[str, Destination] = {}
    routes = _routes(root["routes"], source, destinations)
    route_ids = tuple(route.id for route in routes)
    terms = _ambiguous_terms(root.get("ambiguous_terms", []), source, route_ids, destinations)

    names = (normalize_text(entry.phrase) for entry in destinations.values())
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return RoutesConfig(
        schema_version=version,
        routes=routes,
        ambiguous_terms=terms,
        destinations=MappingProxyType(destinations),
        domain_names=tuple(sorted({name for name in names if _DOTTED.search(name)})),
        source=source,
        sha256=sha256_hex(canonical),
    )


def _routes(value: object, source: str, destinations: dict[str, Destination]) -> tuple[Route, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{source}: routes must be a non-empty list")
    routes: list[Route] = []
    for index, raw in enumerate(value):
        where = f"{source}: routes[{index}]"
        entry = _mapping(raw, where)
        _check_keys(entry, where, _ROUTE_KEYS, _ROUTE_KEYS)
        route_id = _route_id(entry["id"], f"{where}.id")
        if any(route.id == route_id for route in routes):
            raise ConfigError(f"{where}: duplicate route id {route_id!r}")
        raw_aliases = entry["aliases"]
        if not isinstance(raw_aliases, list) or not raw_aliases:
            raise ConfigError(f"{where}.aliases must be a non-empty list")
        aliases: list[str] = []
        for number, raw_alias in enumerate(raw_aliases):
            alias_where = f"{where}.aliases[{number}]"
            alias = _phrase(raw_alias, alias_where)
            _register(destinations, Destination(alias, (route_id,)), alias_where)
            aliases.append(alias)
        routes.append(Route(route_id, tuple(aliases)))
    return tuple(routes)


def _ambiguous_terms(
    value: object, source: str, route_ids: tuple[str, ...], destinations: dict[str, Destination]
) -> tuple[AmbiguousTerm, ...]:
    if not isinstance(value, list):
        raise ConfigError(f"{source}: ambiguous_terms must be a list")
    terms: list[AmbiguousTerm] = []
    for index, raw in enumerate(value):
        where = f"{source}: ambiguous_terms[{index}]"
        entry = _mapping(raw, where)
        _check_keys(entry, where, _TERM_KEYS, _TERM_KEYS)
        term = _phrase(entry["term"], f"{where}.term")
        named = _term_route_ids(entry["routes"], f"{where}.routes", route_ids)
        _register(destinations, Destination(term, named), where)
        terms.append(AmbiguousTerm(term, named))
    return tuple(terms)


def _term_route_ids(value: object, where: str, route_ids: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) < 2:
        raise ConfigError(f"{where} must list at least two route ids")
    named: list[str] = []
    for number, raw in enumerate(value):
        route_id = _route_id(raw, f"{where}[{number}]")
        if route_id not in route_ids:
            raise ConfigError(
                f"{where}[{number}]: {route_id!r} is not a configured route; "
                f"configured: {', '.join(route_ids)}"
            )
        if route_id in named:
            raise ConfigError(f"{where}[{number}]: duplicate route id {route_id!r}")
        named.append(route_id)
    return tuple(route_id for route_id in route_ids if route_id in named)  # configuration order


def _route_id(value: object, where: str) -> str:
    route_id = _text(value, where)
    if not _ROUTE_ID.fullmatch(route_id):
        raise ConfigError(f"{where}: {route_id!r} is not a semantic route id such as 'project_qmind'")
    return route_id


def _phrase(value: object, where: str) -> str:
    phrase = _text(value, where)
    if not _PHRASE.fullmatch(phrase):
        raise ConfigError(
            f"{where}: {phrase!r} must be a plain name: letters and digits separated by "
            "spaces, periods, hyphens, or apostrophes, never a URL or path"
        )
    normalized = normalize_text(phrase)
    if _DOTTED.search(normalized) and not _BARE_DOMAIN.fullmatch(normalized):
        raise ConfigError(
            f"{where}: {phrase!r} has a period inside a word, which only a bare domain name "
            "such as 'example.dev' may have"
        )
    return phrase


def _register(destinations: dict[str, Destination], entry: Destination, where: str) -> None:
    """Add ``entry`` under its phrase key; reject phrases that are unusable or collide."""
    tokens = tokenize(entry.phrase)
    joining = sorted(CONNECTORS.intersection(tokens))
    if joining:
        raise ConfigError(
            f"{where}: {entry.phrase!r} contains {joining[0]!r}, which joins two destinations "
            "in a command"
        )
    words = core_words(tokens)
    if not words:
        raise ConfigError(f"{where}: {entry.phrase!r} has no words")
    if len(entry.route_ids) == 1 and set(words) <= GENERIC_WORDS:
        raise ConfigError(
            f"{where}: alias {entry.phrase!r} is too generic to name one route; "
            "list it under ambiguous_terms if it names several"
        )
    key = "".join(words)
    existing = destinations.get(key)
    if existing is not None:
        relation = "duplicates" if existing.route_ids == entry.route_ids else "collides with"
        raise ConfigError(
            f"{where}: {entry.phrase!r} ({', '.join(entry.route_ids)}) {relation} "
            f"{existing.phrase!r} ({', '.join(existing.route_ids)}): both normalize to {key!r}"
        )
    destinations[key] = entry
