-- 激活高维向量插件
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. 统一原子能力树 (严格替代旧有的 dimensions/sub_skills)
CREATE TABLE ability_taxonomy_nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ability_code VARCHAR(64) UNIQUE NOT NULL,
    ability_name VARCHAR(100) NOT NULL,
    layer SMALLINT NOT NULL CHECK (layer IN (32, 128, 1024)),
    vector_index INTEGER NOT NULL,
    status VARCHAR(20) DEFAULT 'active'
);

CREATE INDEX idx_ability_layer ON ability_taxonomy_nodes(layer);

-- 2. 考核快照账本 (事件溯源，替代 update 覆盖)
CREATE TABLE question_ability_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id UUID NOT NULL,
    candidate_id UUID NOT NULL,
    ability_id UUID NOT NULL REFERENCES ability_taxonomy_nodes(id),
    raw_score NUMERIC(5,2) NOT NULL CHECK (raw_score BETWEEN 0 AND 100),
    normalized_score NUMERIC(6,5) NOT NULL,
    weight NUMERIC(6,5) NOT NULL,
    contribution_score NUMERIC(8,5) GENERATED ALWAYS AS (normalized_score * weight) STORED,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_qas_assessment ON question_ability_scores(assessment_id);
CREATE INDEX idx_qas_candidate ON question_ability_scores(candidate_id);

-- 3. 候选人资产主表 (三层分级高维向量)
CREATE TABLE candidate_vectors (
    candidate_id UUID PRIMARY KEY,
    -- JSONB 存储稀疏矩阵与 Cross-Encoder 弹药
    profile_data JSONB NOT NULL DEFAULT '{"verified_skills": [], "reranker_payload": ""}'::jsonb,
    
    -- 三层召回向量
    ability_vec_32 vector(32),
    ability_vec_128 vector(128),
    ability_vec_1024 vector(1024),
    
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4. 极致搜索索引构建
-- 稀疏层：使用 GIN 索引加速 BM25/文本匹配
CREATE INDEX idx_cand_profile_gin ON candidate_vectors USING GIN (profile_data);

-- 稠密层：强制采用 HNSW 索引加速余弦相似度计算 (ef_construction 和 m 根据内存调优)
CREATE INDEX idx_vec_32_hnsw ON candidate_vectors USING hnsw (ability_vec_32 vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX idx_vec_128_hnsw ON candidate_vectors USING hnsw (ability_vec_128 vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX idx_vec_1024_hnsw ON candidate_vectors USING hnsw (ability_vec_1024 vector_cosine_ops) WITH (m = 24, ef_construction = 100);
