-- Canonical sync schema: the desktop migration ladder's endpoint shape
-- (user_version 11), pinned by the desktop drift-guard test.
-- Generated from a freshly migrated desktop database; kept in lockstep by the
-- desktop drift-guard test. Parents-first (merge-registry order); meta rides
-- along; the FTS5 table is intentionally absent (never merged).

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    display_name TEXT NOT NULL,
    email TEXT UNIQUE,
    avatar_type TEXT DEFAULT 'text',
    avatar_bg TEXT,
    avatar_text TEXT,
    avatar_image_path TEXT,
    avatar_image_url TEXT,
    is_self INTEGER DEFAULT 0,
    status TEXT DEFAULT 'active',
    last_seen_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS user_relationships (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    related_user_id TEXT NOT NULL,
    relationship_type TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (related_user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE(user_id, related_user_id)
);

CREATE TABLE IF NOT EXISTS providers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    provider_type TEXT NOT NULL,
    api_key TEXT,
    base_url TEXT,
    api_style TEXT,
    description TEXT,
    is_enabled INTEGER DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    provider_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    description TEXT,
    is_starred INTEGER DEFAULT 0,
    is_deleted INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT,
    FOREIGN KEY (provider_id) REFERENCES providers(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS model_parameter_presets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    temperature REAL,
    max_tokens INTEGER,
    top_p REAL,
    frequency_penalty REAL,
    presence_penalty REAL,
    additional_params TEXT,
    is_system INTEGER DEFAULT 0,
    is_default INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS assistants (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    role TEXT,
    description TEXT,
    system_prompt TEXT NOT NULL,
    user_prompt TEXT,
    model_id TEXT NOT NULL,
    model_parameter_preset_id TEXT,
    avatar_type TEXT DEFAULT 'text',
    avatar_bg TEXT,
    avatar_text TEXT,
    avatar_image_path TEXT,
    avatar_image_url TEXT,
    group_name TEXT,
    is_starred INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT,
    FOREIGN KEY (model_id) REFERENCES models(id),
    FOREIGN KEY (model_parameter_preset_id) REFERENCES model_parameter_presets(id)
);

CREATE TABLE IF NOT EXISTS tools (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    endpoint TEXT,
    config TEXT,
    description TEXT,
    is_enabled INTEGER DEFAULT 1,
    auth_token TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS skills (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    source TEXT NOT NULL DEFAULT 'user',
    path TEXT NOT NULL,
    icon TEXT,
    required_tool_ids TEXT,
    allow_model_invocation INTEGER DEFAULT 1,
    allow_user_invocation INTEGER DEFAULT 1,
    content_hash TEXT,
    cached_instructions TEXT,
    is_enabled INTEGER DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT,
    UNIQUE(name, source)
);

CREATE TABLE IF NOT EXISTS knowledge_bases (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    content TEXT,
    url TEXT,
    metadata TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS prompts (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    content TEXT NOT NULL,
    description TEXT,
    category TEXT,
    is_system INTEGER DEFAULT 0,
    is_starred INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS fetch_results (
    id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL DEFAULT 'search',
    source_id TEXT,
    url TEXT NOT NULL,
    title TEXT,
    description TEXT,
    storage_path TEXT NOT NULL,
    content_type TEXT NOT NULL,
    original_mime TEXT,
    status TEXT DEFAULT 'pending',
    error TEXT,
    keywords TEXT,
    headings TEXT,
    original_size INTEGER,
    processed_size INTEGER,
    favicon_url TEXT,
    content_hash TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS files (
    id TEXT PRIMARY KEY,
    file_name TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    mime_type TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    content_hash TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT '',
    deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT,
    sender_type TEXT NOT NULL,
    sender_id TEXT,
    content TEXT NOT NULL,
    tokens INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT '',
    deleted_at TEXT,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS search_results (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    query TEXT NOT NULL,
    engine TEXT NOT NULL,
    total_results INTEGER,
    display_order INTEGER DEFAULT 0,
    searched_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT '',
    deleted_at TEXT,
    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS search_decisions (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    reasoning TEXT NOT NULL,
    search_needed INTEGER NOT NULL,
    search_query TEXT,
    search_result_id TEXT,
    display_order INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT '',
    deleted_at TEXT,
    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE,
    FOREIGN KEY (search_result_id) REFERENCES search_results(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS thinking_steps (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT DEFAULT 'llm',
    display_order INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT '',
    deleted_at TEXT,
    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id TEXT PRIMARY KEY,
    call_id TEXT,
    message_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    tool_input TEXT,
    tool_output TEXT,
    status TEXT DEFAULT 'pending',
    error TEXT,
    duration_ms INTEGER,
    display_order INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    updated_at TEXT NOT NULL DEFAULT '',
    deleted_at TEXT,
    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS code_executions (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    language TEXT NOT NULL,
    code TEXT NOT NULL,
    output TEXT,
    exit_code INTEGER,
    status TEXT DEFAULT 'pending',
    error TEXT,
    duration_ms INTEGER,
    display_order INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    updated_at TEXT NOT NULL DEFAULT '',
    deleted_at TEXT,
    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS compactions (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    summary TEXT NOT NULL DEFAULT '',
    first_kept_message_id TEXT NOT NULL,
    tokens_before INTEGER NOT NULL DEFAULT 0,
    method TEXT NOT NULL DEFAULT 'soft',
    archive_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT '',
    deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS content_blocks (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    content TEXT NOT NULL,
    display_order INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT '',
    deleted_at TEXT,
    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS message_attachments (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    attachment_type TEXT NOT NULL CHECK(attachment_type IN ('file')),
    attachment_id TEXT NOT NULL,
    display_order INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT '',
    deleted_at TEXT,
    UNIQUE(message_id, attachment_id),
    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS message_contexts (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    context_type TEXT NOT NULL CHECK(context_type IN ('fetch_result')),
    context_id TEXT NOT NULL,
    display_order INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT '',
    deleted_at TEXT,
    UNIQUE(message_id, context_type, context_id),
    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS message_prompts (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    prompt_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT '',
    deleted_at TEXT,
    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE,
    FOREIGN KEY (prompt_id) REFERENCES prompts(id) ON DELETE CASCADE,
    UNIQUE(message_id, prompt_id)
);

CREATE TABLE IF NOT EXISTS message_knowledge_bases (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    knowledge_base_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT '',
    deleted_at TEXT,
    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE,
    FOREIGN KEY (knowledge_base_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    UNIQUE(message_id, knowledge_base_id)
);

CREATE TABLE IF NOT EXISTS message_tools (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    tool_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT '',
    deleted_at TEXT,
    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE,
    FOREIGN KEY (tool_id) REFERENCES tools(id) ON DELETE CASCADE,
    UNIQUE(message_id, tool_id)
);

CREATE TABLE IF NOT EXISTS assistant_tools (
    id TEXT PRIMARY KEY,
    assistant_id TEXT NOT NULL,
    tool_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT '',
    deleted_at TEXT,
    FOREIGN KEY (assistant_id) REFERENCES assistants(id) ON DELETE CASCADE,
    FOREIGN KEY (tool_id) REFERENCES tools(id) ON DELETE CASCADE,
    UNIQUE(assistant_id, tool_id)
);

CREATE TABLE IF NOT EXISTS assistant_skills (
    id TEXT PRIMARY KEY,
    assistant_id TEXT NOT NULL,
    skill_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT '',
    deleted_at TEXT,
    FOREIGN KEY (assistant_id) REFERENCES assistants(id) ON DELETE CASCADE,
    FOREIGN KEY (skill_id) REFERENCES skills(id) ON DELETE CASCADE,
    UNIQUE(assistant_id, skill_id)
);

CREATE TABLE IF NOT EXISTS assistant_knowledge_bases (
    id TEXT PRIMARY KEY,
    assistant_id TEXT NOT NULL,
    knowledge_base_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT '',
    deleted_at TEXT,
    FOREIGN KEY (assistant_id) REFERENCES assistants(id) ON DELETE CASCADE,
    FOREIGN KEY (knowledge_base_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    UNIQUE(assistant_id, knowledge_base_id)
);

CREATE TABLE IF NOT EXISTS conversation_settings (
    conversation_id TEXT PRIMARY KEY,
    use_provider_defaults INTEGER DEFAULT 1,
    use_custom_parameters INTEGER DEFAULT 0,
    parameter_overrides TEXT,
    context_message_count INTEGER,
    selected_preset_id TEXT,
    system_prompt_mode TEXT DEFAULT 'none',
    selected_system_prompt_id TEXT,
    custom_system_prompt TEXT,
    user_prompt_mode TEXT DEFAULT 'none',
    selected_user_prompt_id TEXT,
    custom_user_prompt TEXT,
    enabled_mcp_server_ids TEXT,
    enabled_skill_ids TEXT,
    working_directory TEXT,
    updated_at TEXT NOT NULL DEFAULT '',
    deleted_at TEXT,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
    FOREIGN KEY (selected_preset_id) REFERENCES model_parameter_presets(id) ON DELETE SET NULL,
    FOREIGN KEY (selected_system_prompt_id) REFERENCES prompts(id) ON DELETE SET NULL,
    FOREIGN KEY (selected_user_prompt_id) REFERENCES prompts(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS conversation_participants (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    participant_type TEXT NOT NULL,
    participant_id TEXT,
    display_name TEXT,
    role TEXT DEFAULT 'member',
    status TEXT DEFAULT 'active',
    joined_at TEXT NOT NULL,
    left_at TEXT,
    last_read_at TEXT,
    metadata TEXT,
    updated_at TEXT NOT NULL DEFAULT '',
    deleted_at TEXT,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
    UNIQUE(conversation_id, participant_type, participant_id)
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_code_executions_message ON code_executions(message_id);
CREATE INDEX IF NOT EXISTS idx_compactions_conversation ON compactions(conversation_id);
CREATE INDEX IF NOT EXISTS idx_content_blocks_message ON content_blocks(message_id);
CREATE INDEX IF NOT EXISTS idx_conversation_participants_conversation ON conversation_participants(conversation_id);
CREATE INDEX IF NOT EXISTS idx_conversation_participants_status ON conversation_participants(conversation_id, status);
CREATE INDEX IF NOT EXISTS idx_fetch_results_content_hash ON fetch_results(content_hash);
CREATE INDEX IF NOT EXISTS idx_fetch_results_source ON fetch_results(source_type, source_id);
CREATE INDEX IF NOT EXISTS idx_files_content_hash ON files(content_hash);
CREATE INDEX IF NOT EXISTS idx_message_attachments_message ON message_attachments(message_id);
CREATE INDEX IF NOT EXISTS idx_message_contexts_message ON message_contexts(message_id);
CREATE INDEX IF NOT EXISTS idx_messages_conversation_created ON messages(conversation_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_search_decisions_message ON search_decisions(message_id);
CREATE INDEX IF NOT EXISTS idx_search_results_message ON search_results(message_id);
CREATE INDEX IF NOT EXISTS idx_thinking_steps_message ON thinking_steps(message_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_message ON tool_calls(message_id);
CREATE INDEX IF NOT EXISTS idx_user_relationships_user ON user_relationships(user_id, relationship_type);
