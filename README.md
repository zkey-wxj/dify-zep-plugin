# Dify Zep Plugin

This plugin exposes Zep Cloud SDK methods as Dify tools.

## Setup

1. Install the plugin in your Dify workspace.
2. Create a Zep project and get an API key.
3. Configure credentials:
   - `zep_api_key` (required)
   - `zep_base_url` (optional, default `https://api.getzep.com`)

## Local Run

Use the project virtual environment:

```bash
.venv/Scripts/python.exe -m main
```

## Tool List

Current tools are mapped one-to-one with SDK methods (no action router).

### User

- `user_add` -> `client.user.add()`
- `user_list_ordered` -> `client.user.list_ordered()`
- `user_get` -> `client.user.get(user_id)`
- `user_update` -> `client.user.update(user_id)`
- `user_delete` -> `client.user.delete(user_id)`
- `user_get_threads` -> `client.user.get_threads(user_id)`

### Thread

- `thread_create` -> `client.thread.create()`
- `thread_get` -> `client.thread.get(thread_id)`
- `thread_delete` -> `client.thread.delete(thread_id)`
- `thread_add_messages` -> `client.thread.add_messages(thread_id)`
- `thread_get_user_context` -> `client.thread.get_user_context(thread_id)`

### Graph

- `graph_add` -> `client.graph.add()`
- `graph_create` -> `client.graph.create()`
- `graph_search` -> `client.graph.search()`
- `graph_node_get` -> `client.graph.node.get(uuid_)`
- `graph_node_get_by_graph_id` -> `client.graph.node.get_by_graph_id(graph_id)`
- `graph_edge_get_by_graph_id` -> `client.graph.edge.get_by_graph_id(graph_id)`
- `graph_episode_get` -> `client.graph.episode.get(uuid_)`
- `graph_delete` -> `client.graph.delete(graph_id)`
- `graph_observation_get` -> `client.graph.observation.get(uuid_)`
- `graph_observation_delete` -> reserved (see limitation below)

## Parameter Conventions

- `metadata_json`: JSON object string.
- `messages_json`: JSON array string, each item must include:
  - `content`
  - `role`
- Optional pagination fields:
  - `page_number`, `page_size`
  - `limit`, `cursor`, `uuid_cursor`

## Known Limitation

- In current installed SDK (`zep-cloud 3.22.0`), `client.graph.observation.delete(uuid_)` is not exposed.
- `graph_observation_delete` currently returns `NotImplementedError` explicitly.
