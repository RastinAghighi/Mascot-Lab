"""Tests for configs/routes.yaml (the navigation allowlist) and phrase normalization."""

import dataclasses
import re
from collections.abc import Callable
from typing import Any

import pytest
import yaml

from mascot_lab import ConfigError
from mascot_lab.paths import KNOWLEDGE_PATH
from mascot_lab.phrases import phrase_key, tokenize
from mascot_lab.routes import RoutesConfig, load_routes_config, parse_routes_config

# The five featured projects (facts.md P01) by their registry route IDs.
FEATURED_PROJECT_ROUTES = (
    "project_qmind",
    "project_ingenuity",
    "project_satira",
    "project_deeplense",
    "project_website",
)


def registry() -> dict[str, str]:
    """Route ID -> "Actual path/URL" cell of the navigation registry in knowledge/facts.md."""
    text = KNOWLEDGE_PATH.read_text(encoding="utf-8")
    section = text.split("## Navigation registry", 1)[1].split("\n## ", 1)[0]
    rows = re.findall(r"^\|\s*`([^`]+)`\s*\|[^|]*\|\s*([^|]*?)\s*\|$", section, flags=re.MULTILINE)
    return dict(rows)


@pytest.fixture(scope="module")
def routes_config() -> RoutesConfig:
    return load_routes_config()


def valid_data() -> dict[str, Any]:
    """A small, valid synthetic allowlist."""
    return {
        "schema_version": 1,
        "routes": [
            {"id": "route_a", "aliases": ["Alpha", "Alpha Lab"]},
            {"id": "route_b", "aliases": ["Beta", "beta.example"]},
        ],
        "ambiguous_terms": [{"term": "Greek", "routes": ["route_b", "route_a"]}],
    }


def dump(data: object) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


# --------------------------------------------------------------- real config


def test_registry_lists_the_known_semantic_route_ids() -> None:
    # Pins this test file's reading of facts.md; a registry change must be deliberate.
    assert list(registry()) == [*FEATURED_PROJECT_ROUTES, "mascot_chat"]


def test_no_registry_destination_has_an_approved_path() -> None:
    assert set(registry().values()) == {"Pending"}


def test_config_routes_exactly_the_five_featured_projects(routes_config: RoutesConfig) -> None:
    assert routes_config.route_ids == FEATURED_PROJECT_ROUTES
    assert set(routes_config.route_ids) <= set(registry())


def test_mascot_chat_has_no_text_alias(routes_config: RoutesConfig) -> None:
    assert "mascot_chat" in registry()
    assert "mascot_chat" not in routes_config.route_ids
    for entry in routes_config.destinations.values():
        assert "mascot_chat" not in entry.route_ids


def test_each_project_is_named_by_its_featured_name(routes_config: RoutesConfig) -> None:
    aliases = {route.id: route.aliases for route in routes_config.routes}
    assert "QMIND" in aliases["project_qmind"]
    assert "Ingenuity Lab" in aliases["project_ingenuity"]
    assert "Satira" in aliases["project_satira"]
    assert "DeepLense" in aliases["project_deeplense"]
    assert {"This website", "rastinaghighi.dev"} <= set(aliases["project_website"])


def test_ambiguous_terms_name_several_routes(routes_config: RoutesConfig) -> None:
    terms = {term.term: term.route_ids for term in routes_config.ambiguous_terms}
    assert terms == {
        "project": FEATURED_PROJECT_ROUTES,
        "research": ("project_qmind", "project_ingenuity"),
    }


def test_config_contains_no_url_or_path(routes_config: RoutesConfig) -> None:
    for entry in routes_config.destinations.values():
        assert not re.search(r"[/\\:@?#%=]", entry.phrase), entry.phrase
    # The one dotted phrase is the domain that facts.md names; it is accepted only bare.
    assert routes_config.domain_names == ("rastinaghighi.dev",)


def test_every_phrase_has_its_own_normalized_key(routes_config: RoutesConfig) -> None:
    phrases = [alias for route in routes_config.routes for alias in route.aliases]
    phrases += [term.term for term in routes_config.ambiguous_terms]
    assert len(routes_config.destinations) == len(phrases)
    assert sorted(entry.phrase for entry in routes_config.destinations.values()) == sorted(phrases)


def test_config_records_a_stable_hash_and_relative_source(routes_config: RoutesConfig) -> None:
    assert re.fullmatch(r"[0-9a-f]{64}", routes_config.sha256)
    assert load_routes_config().sha256 == routes_config.sha256
    assert routes_config.source == "configs/routes.yaml"


def test_loaded_config_is_read_only(routes_config: RoutesConfig) -> None:
    with pytest.raises(TypeError):
        routes_config.destinations["evil"] = routes_config.destinations["qmind"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        routes_config.routes[0].id = "project_evil"


# ----------------------------------------------------------- normalization


@pytest.mark.parametrize(
    ("phrase", "key"),
    [
        ("QMIND", "qmind"),
        ("the Q-MIND page", "qmind"),
        ("Rastin's QMIND project", "qmind"),
        ("ＱＭＩＮＤ", "qmind"),  # full-width letters
        ("Q​MIND", "qmind"),  # zero-width space
        ("Satíra", "satira"),  # accent
        ("Deep_Lense", "deeplense"),
        ("rastinaghighi.dev", "rastinaghighidev"),
        ("the page for DeepLense", "deeplense"),
        ("the project Satira", "satira"),
        ("the project page", "project"),
        ("the research project", "research"),
        ("QMIND mechanistic interpretability work", "qmindmechanisticinterpretability"),
        ("Satire", "satire"),  # spelling is never corrected
    ],
)
def test_phrase_key_ignores_form_but_not_spelling(phrase: str, key: str) -> None:
    assert phrase_key(tokenize(phrase)) == key


def test_tokenize_keeps_the_marks_that_join_destinations() -> None:
    assert tokenize("QMIND, Satira & DeepLense!") == ("qmind", ",", "satira", "&", "deeplense")
    assert tokenize("let's see QMIND's page") == ("let", "s", "see", "qmind", "s", "page")


# ---------------------------------------------------------- valid synthetic


def test_valid_config_file_loads(tmp_path) -> None:
    path = tmp_path / "routes.yaml"
    path.write_text(dump(valid_data()), encoding="utf-8")
    config = load_routes_config(path)
    assert config.route_ids == ("route_a", "route_b")
    assert config.routes[0].aliases == ("Alpha", "Alpha Lab")
    assert set(config.destinations) == {"alpha", "alphalab", "beta", "betaexample", "greek"}
    assert config.destinations["alphalab"].route_ids == ("route_a",)
    assert config.domain_names == ("beta.example",)


def test_ambiguous_term_routes_follow_the_configuration_order() -> None:
    (term,) = parse_routes_config(dump(valid_data())).ambiguous_terms
    assert term.route_ids == ("route_a", "route_b")


def test_ambiguous_terms_are_optional() -> None:
    data = valid_data()
    del data["ambiguous_terms"]
    assert parse_routes_config(dump(data)).ambiguous_terms == ()


def test_a_generic_word_may_be_an_ambiguous_term() -> None:
    data = valid_data()
    data["ambiguous_terms"] = [{"term": "project", "routes": ["route_a", "route_b"]}]
    assert parse_routes_config(dump(data)).destinations["project"].route_ids == ("route_a", "route_b")


def test_hash_ignores_comments_and_layout_but_tracks_values() -> None:
    text = dump(valid_data())
    base = parse_routes_config(text)
    relaid = parse_routes_config("# a comment\n" + text.replace("- Beta\n", "-   Beta  # note\n"))
    changed = parse_routes_config(text.replace("- Beta\n", "- Gamma\n"))
    assert relaid.sha256 == base.sha256
    assert changed.sha256 != base.sha256


# -------------------------------------------------------- invalid synthetic


def _root(**changes: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return lambda data: {**data, **changes}


def _without(key: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return lambda data: {k: v for k, v in data.items() if k != key}


def _route(index: int, **changes: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def mutate(data: dict[str, Any]) -> dict[str, Any]:
        data["routes"][index].update(changes)
        return data

    return mutate


def _route_without(index: int, key: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def mutate(data: dict[str, Any]) -> dict[str, Any]:
        del data["routes"][index][key]
        return data

    return mutate


def _term(index: int, **changes: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def mutate(data: dict[str, Any]) -> dict[str, Any]:
        data["ambiguous_terms"][index].update(changes)
        return data

    return mutate


def _twice(key: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return lambda data: {**data, key: data[key] * 2}


INVALID_STRUCTURES = [
    pytest.param(lambda data: [data], "root must be a mapping", id="root-is-a-list"),
    pytest.param(_root(paths={"route_a": "/alpha"}), "unknown key.*paths", id="paths-section"),
    pytest.param(_without("routes"), "missing required key.*routes", id="missing-routes"),
    pytest.param(_root(schema_version=2), "schema_version", id="wrong-schema-version"),
    pytest.param(_root(routes=[]), "routes must be a non-empty list", id="empty-routes"),
    pytest.param(_root(routes={"id": "route_a"}), "routes must be a non-empty list", id="routes-mapping"),
    pytest.param(_root(routes=["route_a"]), "must be a mapping", id="route-not-a-mapping"),
    # No paths or URLs: the schema has no place for them.
    pytest.param(_route(0, url="https://alpha.example"), "unknown key.*url", id="route-with-url"),
    pytest.param(_route(0, path="/alpha"), "unknown key.*path", id="route-with-path"),
    pytest.param(_route_without(0, "id"), "missing required key.*id", id="missing-id"),
    pytest.param(_route_without(0, "aliases"), "missing required key.*aliases", id="missing-aliases"),
    pytest.param(_route(0, id="Route_A"), "semantic route id", id="uppercase-id"),
    pytest.param(_route(0, id="https://alpha.example"), "semantic route id", id="url-as-id"),
    pytest.param(_route(0, id="route a"), "semantic route id", id="id-with-space"),
    pytest.param(_route(0, id=7), "id must be a non-empty string", id="numeric-id"),
    pytest.param(_route(1, id="route_a"), "duplicate route id", id="duplicate-id"),
    pytest.param(_route(0, aliases=[]), "aliases must be a non-empty list", id="no-aliases"),
    pytest.param(_route(0, aliases="Alpha"), "aliases must be a non-empty list", id="aliases-string"),
    pytest.param(_route(0, aliases=[12]), "must be a non-empty string", id="numeric-alias"),
    pytest.param(_route(0, aliases=[True]), "must be a non-empty string", id="boolean-alias"),
    pytest.param(_route(0, aliases=["  "]), "must be a non-empty string", id="blank-alias"),
    pytest.param(_route(0, aliases=[" Alpha"]), "whitespace", id="padded-alias"),
    pytest.param(_route(0, aliases=["https://alpha.example"]), "plain name", id="url-alias"),
    pytest.param(_route(0, aliases=["alpha/page"]), "plain name", id="path-alias"),
    pytest.param(_route(0, aliases=["alpha@example.com"]), "plain name", id="email-alias"),
    pytest.param(_route(0, aliases=["alpha.example/x"]), "plain name", id="domain-with-path-alias"),
    pytest.param(_route(0, aliases=["Alpha!"]), "plain name", id="punctuation-alias"),
    pytest.param(_route(0, aliases=["Alpha v1.2"]), "period inside a word", id="dotted-non-domain"),
    pytest.param(_route(0, aliases=["Alpha and Omega"]), "joins two destinations", id="connector"),
    pytest.param(_route(0, aliases=["the page"]), "too generic", id="generic-alias"),
    pytest.param(_route(0, aliases=["Rastin's project"]), "too generic", id="generic-possessive"),
    # Collisions: phrases that normalize to the same key, within or across routes.
    pytest.param(_route(0, aliases=["Alpha", "ALPHA"]), "duplicates", id="duplicate-by-case"),
    pytest.param(_route(0, aliases=["Alpha Lab", "alpha-lab"]), "duplicates", id="duplicate-by-hyphen"),
    pytest.param(_route(0, aliases=["Alpha", "the Alpha page"]), "duplicates", id="duplicate-by-filler"),
    pytest.param(_route(1, aliases=["Alpha"]), "collides with", id="collision-same-text"),
    pytest.param(_route(1, aliases=["AL PHA"]), "collides with", id="collision-by-spacing"),
    pytest.param(_route(1, aliases=["Álpha"]), "collides with", id="collision-by-accent"),
    pytest.param(_route(1, aliases=["Alpha project"]), "collides with", id="collision-by-filler"),
    pytest.param(_term(0, term="Alpha"), "collides with", id="term-collides-with-alias"),
    pytest.param(_twice("ambiguous_terms"), "duplicates", id="duplicate-term"),
    # Ambiguous terms.
    pytest.param(_root(ambiguous_terms={"term": "Greek"}), "must be a list", id="terms-mapping"),
    pytest.param(_term(0, routes=["route_a"]), "at least two route ids", id="term-one-route"),
    pytest.param(_term(0, routes="route_a"), "at least two route ids", id="term-routes-string"),
    pytest.param(_term(0, routes=["route_a", "route_c"]), "not a configured route", id="term-unknown"),
    pytest.param(_term(0, routes=["route_a", "route_a"]), "duplicate route id", id="term-repeat"),
    pytest.param(_term(0, term="https://greek.example"), "plain name", id="url-term"),
    pytest.param(_term(0, url="https://greek.example"), "unknown key.*url", id="term-with-url"),
]


@pytest.mark.parametrize(("mutate", "message"), INVALID_STRUCTURES)
def test_invalid_structure_fails_closed(
    mutate: Callable[[dict[str, Any]], object], message: str
) -> None:
    text = dump(mutate(valid_data()))
    with pytest.raises(ConfigError, match=message):
        parse_routes_config(text)


VALID_YAML = dump(valid_data())

INVALID_YAML = [
    pytest.param("", "root must be a mapping", id="empty-file"),
    pytest.param("routes: [\n", "invalid YAML", id="syntax-error"),
    pytest.param(VALID_YAML + "---\nschema_version: 1\n", "invalid YAML", id="two-documents"),
    pytest.param(
        VALID_YAML.replace("- id: route_a\n", "- id: route_a\n  id: route_b\n"),
        "duplicate key 'id'",
        id="duplicate-key",
    ),
    pytest.param(
        "shared: &shared\n  id: route_a\n" + VALID_YAML.replace("- id: route_a\n", "- <<: *shared\n"),
        "merge keys",
        id="merge-key",
    ),
]


@pytest.mark.parametrize(("text", "message"), INVALID_YAML)
def test_invalid_yaml_fails_closed(text: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_routes_config(text)


def test_missing_config_file_fails_closed(tmp_path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_routes_config(tmp_path / "routes.yaml")


def test_non_utf8_config_file_fails_closed(tmp_path) -> None:
    path = tmp_path / "routes.yaml"
    path.write_bytes(dump(valid_data()).encode("utf-8") + b"\xff")
    with pytest.raises(ConfigError, match="UTF-8"):
        load_routes_config(path)
