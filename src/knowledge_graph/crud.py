import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

from knowledge_graph.connection import Neo4jConnection


def _assert_safe_identifier(name: str, what: str) -> None:
    """Prevent injection by restricting to alphanumeric + underscore, starting with letter/underscore."""
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        raise ValueError(
            f"Invalid {what} `{name}`: must match /^[A-Za-z_][A-Za-z0-9_]*$/."
        )


class GraphCrud:
    """
    Robust Neo4j CRUD helper:
      - session management (_execute_read/_execute_write)
      - safe label & relationship interpolation
      - batch methods
      - undirected relationships support
      - vector index management
    """

    def __init__(self, neo4j_connection: Neo4jConnection):
        self._driver = neo4j_connection.get_driver()

    def close(self) -> None:
        """Close the Neo4j driver."""
        self._driver.close()

    # ─── Internal Executors ────────────────────────────────────────────────

    def _execute_read(self, cypher: str, **params) -> List[Dict[str, Any]]:
        """Run a read-only query and return a list of dicts."""
        try:
            with self._driver.session() as session:
                result = session.run(cypher, **params)
                records = result.data()  # fetched inside session
            print(f"Read {len(records)} rows. Cypher={cypher}")
            return records
        except Neo4jError as e:
            print(f"Read failed: {e}\nCypher={cypher}\nParams={params}", exc_info=True)
            raise

    def _execute_write(self, work) -> Any:
        """
        Run write_transaction, passing work(tx)->T, and return its result.
        Catches & logs Neo4jError.
        """
        try:
            with self._driver.session() as session:
                return session.write_transaction(work)
        except Neo4jError as e:
            print(f"Write failed: {e}", exc_info=True)
            raise

    # ─── Node Operations ─────────────────────────────────────────────────

    def create_node(self, label: str, properties: Dict[str, Any]) -> int:
        """Create one node and return its elementId."""
        _assert_safe_identifier(label, "label")
        if not isinstance(properties, dict):
            raise ValueError("Properties must be a dict.")

        def _work(tx):
            cy = f"CREATE (n:`{label}`) SET n = $props RETURN elementId(n) AS id"
            rec = tx.run(cy, props=properties).single()
            if not rec:
                raise RuntimeError(f"Failed to create node {label}")
            return rec["id"]

        node_id = self._execute_write(_work)
        print(f"Created node `{label}` id={node_id}")
        return node_id

    def update_node(self, node_id: int, properties: Dict[str, Any]) -> int:
        """Merge properties into existing node. Returns # of properties set."""
        if not properties:
            return 0

        def _work(tx):
            cy = """
                MATCH (n)
                WHERE elementId(n) = $node_id
                SET n += $props
                RETURN size(keys($props)) AS updated
            """
            rec = tx.run(cy, node_id=node_id, props=properties).single()
            return rec["updated"] if rec else 0

        updated = self._execute_write(_work)
        print(f"Updated node id={node_id}: {updated} props")
        return updated

    def delete_node(self, node_id: int) -> None:
        """Detach-delete a node by elementId."""

        def _work(tx):
            cy = "MATCH (n) WHERE elementId(n) = $id DETACH DELETE n"
            tx.run(cy, id=node_id)

        self._execute_write(_work)
        print(f"Deleted node id={node_id}")

    # ─── Relationship Operations ────────────────────────────────────────

    def create_relationship(
        self,
        from_id: int,
        to_id: Optional[int] = None,
        to_label: Optional[str] = None,
        to_prop_key: Optional[str] = None,
        to_prop_val: Optional[Any] = None,
        rel_type: str = "",
        properties: Optional[Dict[str, Any]] = None,
        undirected: bool = False,
    ) -> List[int]:
        """
        Create a (directed) relationship rel_type from from_id to to_id (or via label/key/val).
        If undirected=True, also creates the reverse.
        Returns list of created relationship elementIds.
        """
        _assert_safe_identifier(rel_type, "relationship type")
        props = properties or {}
        created: List[int] = []

        def _make_cypher(reverse: bool) -> str:
            dir_l = "<" if reverse else ""
            dir_r = ">" if reverse else ""
            if to_id is not None:
                return (
                    f"MATCH (a),(b) "
                    f"WHERE elementId(a)=$from_id AND elementId(b)=$to_id "
                    f"CREATE (a){dir_l}-[r:`{rel_type}` $props]-{dir_r}(b) "
                    f"RETURN elementId(r) AS id"
                )
            else:
                _assert_safe_identifier(to_label, "label")
                _assert_safe_identifier(to_prop_key, "property key")
                return (
                    f"MATCH (a), (b:`{to_label}`) "
                    f"WHERE elementId(a)=$from_id AND b.`{to_prop_key}`=$to_prop_val "
                    f"CREATE (a){dir_l}-[r:`{rel_type}` $props]-{dir_r}(b) "
                    f"RETURN elementId(r) AS id"
                )

        def _work(tx):
            # forward
            rec = tx.run(
                _make_cypher(reverse=False),
                from_id=from_id,
                to_id=to_id,
                to_prop_val=to_prop_val,
                props=props,
            ).single()
            if not rec:
                raise RuntimeError(
                    f"Failed to create {rel_type} from {from_id} to {to_id or to_prop_val}"
                )
            created.append(rec["id"])

            # reverse?
            if undirected:
                rec2 = tx.run(
                    _make_cypher(reverse=True),
                    from_id=from_id,
                    to_id=to_id,
                    to_prop_val=to_prop_val,
                    props=props,
                ).single()
                if rec2:
                    created.append(rec2["id"])

            return created

        ids = self._execute_write(_work)
        print(f"Created rel(s) `{rel_type}`: {ids}")
        return ids

    def create_relationships_batch(
        self, rels_data: List[Dict[str, Any]], undirected: bool = False
    ) -> int:
        """
        Batch create relationships of possibly multiple types.
        Each item: {from_id, to_id, rel_type, properties?}.
        undirected=True mirrors each.
        Returns total relationships created.
        """
        if not rels_data:
            return 0

        total_created = 0
        by_type = defaultdict(list)
        for r in rels_data:
            rt = r.get("rel_type")
            _assert_safe_identifier(rt, "relationship type")
            by_type[rt].append(r)

        def _work(tx):
            cnt = 0
            for rt, items in by_type.items():
                # forward
                cy = (
                    "UNWIND $rows AS row\n"
                    "MATCH (a),(b)\n"
                    " WHERE elementId(a)=row.from_id AND elementId(b)=row.to_id\n"
                    f"CREATE (a)-[r:`{rt}`]->(b)\n"
                    "SET r = coalesce(row.properties, {})\n"
                )
                summ = tx.run(cy, rows=items).consume()
                cnt += summ.counters.relationships_created

                if undirected:
                    # mirrored
                    cy2 = (
                        "UNWIND $rows AS row\n"
                        "MATCH (a),(b)\n"
                        " WHERE elementId(a)=row.to_id AND elementId(b)=row.from_id\n"
                        f"CREATE (a)-[r:`{rt}`]->(b)\n"
                        "SET r = coalesce(row.properties, {})\n"
                    )
                    summ2 = tx.run(cy2, rows=items).consume()
                    cnt += summ2.counters.relationships_created
            return cnt

        created = self._execute_write(_work)
        print(f"Batch created {created} rels{' (undirected)' if undirected else ''}")
        return created

    # ─── Batch Node Operations ─────────────────────────────────────────────

    def create_nodes_batch(
        self, label: str, properties_list: List[Dict[str, Any]]
    ) -> List[int]:
        """Batch-create nodes of one label, returning their elementIds."""
        _assert_safe_identifier(label, "label")
        if not properties_list:
            return []

        def _work(tx):
            cy = (
                "UNWIND $props_list AS props\n"
                f"CREATE (n:`{label}`)\n"
                "SET n = props\n"
                "RETURN elementId(n) AS id"
            )
            return [rec["id"] for rec in tx.run(cy, props_list=properties_list)]

        ids = self._execute_write(_work)
        print(f"Batch created {len(ids)} `{label}` nodes")
        return ids

    def set_node_vector_properties_batch(
        self, vectors_data: List[Dict[str, Any]], property_name: str = "embedding"
    ) -> None:
        """
        Batch-set vector property using Neo4j's vector procedure.
        vectors_data: [{node_id, embedding}, ...]
        """
        if not vectors_data:
            return

        def _work(tx):
            cy = (
                "UNWIND $vecs AS v\n"
                "MATCH (n) WHERE elementId(n)=v.node_id\n"
                "CALL db.create.setNodeVectorProperty(n, $prop, v.embedding)\n"
            )
            return tx.run(cy, vecs=vectors_data, prop=property_name).consume()

        summ = self._execute_write(_work)
        print(f"Vector batch-set: {summ.counters.properties_set} properties")

    # ─── Index & Read ───────────────────────────────────────────────────────

    def ensure_vector_index(
        self,
        index_name: str,
        label: str,
        property_name: str,
        dimensions: Optional[int] = None,
        similarity_function: Optional[str] = None,
    ) -> None:
        """
        CREATE VECTOR INDEX IF NOT EXISTS ... for (n:Label) ON (n.prop)
        Optional indexConfig settings.
        """
        _assert_safe_identifier(index_name, "index name")
        _assert_safe_identifier(label, "label")
        opts = []
        if dimensions:
            opts.append(f"`vector.dimensions`: {dimensions}")
        if similarity_function:
            opts.append(f"`vector.similarity_function`: '{similarity_function}'")
        opts_str = (
            f" OPTIONS {{ indexConfig: {{ {', '.join(opts)} }} }}" if opts else ""
        )

        cy = (
            f"CREATE VECTOR INDEX `{index_name}` IF NOT EXISTS "
            f"FOR (n:`{label}`) ON (n.`{property_name}`){opts_str}"
        )
        # no return value
        self._execute_write(lambda tx: tx.run(cy))
        print(f"Ensured vector index `{index_name}` on {label}({property_name})")

    def get_nodes_with_property(
        self, label: str, property_name: str
    ) -> List[Dict[str, Any]]:
        """Return [{id, property_name: value}, ...] for all nodes where prop is not null."""
        _assert_safe_identifier(label, "label")
        cy = (
            "MATCH (n:`{}`)\n"
            "WHERE n.`{}` IS NOT NULL\n"
            "RETURN elementId(n) AS id, n.`{}` AS `{}`"
        ).format(label, property_name, property_name, property_name)
        return self._execute_read(cy)

    def run_query(self, cypher: str, **params) -> List[Dict[str, Any]]:
        """Expose ad-hoc read-only Cypher."""
        return self._execute_read(cypher, **params)
