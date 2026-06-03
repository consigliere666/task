import asyncio
from pydantic import BaseModel
from typing import List, Dict

# --- 模拟请求体 ---
class SearchRequest(BaseModel):
    requirement_profile_id: str
    query_text: str
    target_vec_32: List[float]
    target_vec_128: List[float]
    target_vec_1024: List[float]
    must_have_skills: List[str]

class CandidateScore(BaseModel):
    candidate_id: str
    score: float

# --- 模拟底层 RPC 调用 (代替连库查询) ---
async def rpc_search_vec32(vec: List[float], limit: int) -> Dict[str, int]:
    return {f"cand_{i}": i for i in range(1, limit + 1)}

async def rpc_search_vec128(vec: List[float], candidates: List[str], limit: int) -> Dict[str, int]:
    return {c: i+1 for i, c in enumerate(candidates[:limit])}

async def rpc_search_vec1024(vec: List[float], candidates: List[str], limit: int) -> Dict[str, int]:
    return {c: i+1 for i, c in enumerate(candidates[:limit])}

async def rpc_sparse_text_search(query: str, limit: int) -> Dict[str, int]:
    return {f"cand_{i*2}": i for i in range(1, limit + 1)}

async def rpc_cross_encoder_rerank(query: str, candidates: List[str]) -> List[CandidateScore]:
    return [CandidateScore(candidate_id=c, score=0.99 - (i*0.01)) for i, c in enumerate(candidates)]

# --- 核心检索管线 ---
async def execute_hybrid_search(req: SearchRequest) -> List[CandidateScore]:
    print("⏳ 开始执行 L1 粗召回 (32维)...")
    l1_ranks = await rpc_search_vec32(req.target_vec_32, limit=200)
    l1_candidates = list(l1_ranks.keys())

    print("⏳ 开始执行 L2 中召回 (128维)...")
    l2_ranks = await rpc_search_vec128(req.target_vec_128, l1_candidates, limit=80)
    l2_candidates = list(l2_ranks.keys())

    print("⏳ 并发执行 L3 精召回 (1024维) 与 GIN 稀疏文本召回...")
    l3_task = rpc_search_vec1024(req.target_vec_1024, l2_candidates, limit=50)
    sparse_task = rpc_sparse_text_search(req.query_text, limit=50)
    l3_ranks, sparse_ranks = await asyncio.gather(l3_task, sparse_task)

    print("🧠 触发 RRF 倒数秩融合算法...")
    RRF_K = 60
    fusion_scores: Dict[str, float] = {}
    all_candidates = set(l3_ranks.keys()).union(set(sparse_ranks.keys()))
    
    for cand in all_candidates:
        dense_rank = l3_ranks.get(cand, 1000) 
        sparse_rank = sparse_ranks.get(cand, 1000)
        score = (1.0 / (RRF_K + dense_rank)) + (1.0 / (RRF_K + sparse_rank))
        fusion_scores[cand] = score

    top_30_candidates = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)[:30]
    top_30_ids = [c[0] for c in top_30_candidates]

    print("🎯 交由 Cross-Encoder 大模型进行深层语义重排...")
    final_ranking = await rpc_cross_encoder_rerank(req.query_text, top_30_ids)
    
    return final_ranking[:3]

# --- 触发测试执行 ---
async def main():
    req = SearchRequest(
        requirement_profile_id="req_001",
        query_text="需要一个能在高并发下处理 Redis 分布式死锁的后端",
        target_vec_32=[0.1] * 32,
        target_vec_128=[0.2] * 128,
        target_vec_1024=[0.3] * 1024,
        must_have_skills=["Golang", "Redis"]
    )
    
    results = await execute_hybrid_search(req)
    
    print("\n✅ 检索链路执行完毕，最终确权输出 Top 3:")
    for rank, res in enumerate(results):
        print(f"  [Top {rank + 1}] 候选人: {res.candidate_id} | 契合度得分: {res.score:.4f}")

# 在 Colab 中直接运行
await main()
