/**
 * NDE (Nondestructive Edition) API Routes
 */
import { Router, Request, Response } from 'express';
import { ULID } from 'ulid';
import crypto from 'crypto';

export const ndeRouter = Router();

// In-memory storage (실제로는 SQLite/PostgreSQL 사용)
const sessions = new Map<string, any>();
const editGraphs = new Map<string, any>();

/**
 * 세션 생성
 * POST /api/nde/session/create
 */
ndeRouter.post('/session/create', async (req: Request, res: Response) => {
  try {
    const { source_uri, policy_id } = req.body;

    if (!source_uri || !policy_id) {
      return res.status(400).json({
        error: 'source_uri and policy_id required'
      });
    }

    // Policy 로드 (Placeholder)
    const policy = await loadPolicy(policy_id);

    // 소스 해시 계산 (Placeholder - 실제로는 S3/파일에서)
    const source_hash = computeHash(source_uri);

    // EditGraph 초기화
    const session_id = ULID();
    const editGraph = {
      graph_id: ULID(),
      source_hash,
      nodes: [],
      created_at: Date.now(),
      merge_metadata: { conflicts: [] }
    };

    sessions.set(session_id, {
      session_id,
      source_uri,
      source_hash,
      policy_id,
      graph_id: editGraph.graph_id,
      created_at: Date.now(),
      status: 'active'
    });

    editGraphs.set(editGraph.graph_id, editGraph);

    res.json({
      session_id,
      graph_id: editGraph.graph_id,
      source_hash,
      policy_id
    });
  } catch (error) {
    res.status(500).json({ error: String(error) });
  }
});

/**
 * 편집 적용 (SSE 스트리밍)
 * POST /api/nde/edit/apply
 */
ndeRouter.post('/edit/apply', async (req: Request, res: Response) => {
  try {
    const { session_id, node } = req.body;

    if (!session_id || !node) {
      return res.status(400).json({
        error: 'session_id and node required'
      });
    }

    const session = sessions.get(session_id);
    if (!session) {
      return res.status(404).json({ error: 'Session not found' });
    }

    // SSE 헤더 설정
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    // Policy Gate 검증
    const policy = await loadPolicy(session.policy_id);
    const [passed, reason] = await policyGateCheck(node, policy);

    if (!passed) {
      res.write(`data: ${JSON.stringify({
        event: 'error',
        reason
      })}\n\n`);
      return res.end();
    }

    // EditGraph에 노드 추가
    const graph = editGraphs.get(session.graph_id);
    if (!graph) {
      res.write(`data: ${JSON.stringify({
        event: 'error',
        reason: 'EditGraph not found'
      })}\n\n`);
      return res.end();
    }

    const node_id = node.node_id || ULID();
    const editNode = {
      ...node,
      node_id,
      timestamp: Date.now()
    };

    graph.nodes.push(editNode);

    // SSE 이벤트 전송
    res.write(`data: ${JSON.stringify({
      event: 'node_added',
      node_id,
      graph_hash: computeGraphHash(graph)
    })}\n\n`);

    // NATS로 preview render 요청 (비동기 - Placeholder)
    // await natsClient.publish('ipc.req.apex-audio.render.preview', {...});

    res.write(`data: ${JSON.stringify({
      event: 'queued',
      node_id
    })}\n\n`);

    // 시뮬레이션: 렌더링 진행 상황
    setTimeout(() => {
      res.write(`data: ${JSON.stringify({
        event: 'rendering',
        progress: 0.5
      })}\n\n`);

      setTimeout(() => {
        res.write(`data: ${JSON.stringify({
          event: 'complete',
          node_id,
          preview_url: `s3://bucket/preview_${node_id}.mp3`
        })}\n\n`);
        res.end();
      }, 500);
    }, 500);

  } catch (error) {
    if (!res.headersSent) {
      res.status(500).json({ error: String(error) });
    } else {
      res.end();
    }
  }
});

/**
 * Preview 렌더
 * POST /api/nde/render/preview
 */
ndeRouter.post('/render/preview', async (req: Request, res: Response) => {
  try {
    const { session_id } = req.body;

    const session = sessions.get(session_id);
    if (!session) {
      return res.status(404).json({ error: 'Session not found' });
    }

    // SSE 헤더
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    const graph = editGraphs.get(session.graph_id);

    res.write(`data: ${JSON.stringify({
      event: 'started',
      graph_id: graph.graph_id,
      node_count: graph.nodes.length
    })}\n\n`);

    // 시뮬레이션: 렌더링 진행
    const steps = ['loading', 'processing', 'qc', 'complete'];
    for (let i = 0; i < steps.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 300));
      res.write(`data: ${JSON.stringify({
        event: steps[i],
        progress: (i + 1) / steps.length
      })}\n\n`);
    }

    res.write(`data: ${JSON.stringify({
      event: 'complete',
      preview_url: `s3://bucket/preview_${session_id}.mp3`
    })}\n\n`);

    res.end();
  } catch (error) {
    if (!res.headersSent) {
      res.status(500).json({ error: String(error) });
    } else {
      res.end();
    }
  }
});

/**
 * Final 렌더 (SSE + QC)
 * POST /api/nde/render/final
 */
ndeRouter.post('/render/final', async (req: Request, res: Response) => {
  try {
    const { session_id } = req.body;

    const session = sessions.get(session_id);
    if (!session) {
      return res.status(404).json({ error: 'Session not found' });
    }

    // SSE 헤더
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    const graph = editGraphs.get(session.graph_id);

    res.write(`data: ${JSON.stringify({
      event: 'started',
      quality: 'high',
      qc_gates: ['technical', 'locality', 'similarity']
    })}\n\n`);

    // NATS로 final render 요청
    const job_id = ULID();

    // 시뮬레이션
    const steps = [
      { event: 'rendering', progress: 0.3 },
      { event: 'qc_technical', progress: 0.6 },
      { event: 'qc_locality', progress: 0.8 },
      { event: 'packaging', progress: 0.95 }
    ];

    for (const step of steps) {
      await new Promise(resolve => setTimeout(resolve, 400));
      res.write(`data: ${JSON.stringify(step)}\n\n`);
    }

    // QC 결과
    const qc_results = {
      technical: { passed: true, lufs: -18.5, true_peak_db: -2.1 },
      locality: { passed: true, out_mask_delta: 0.02 },
      similarity: { passed: true, max_similarity: 0.72 }
    };

    res.write(`data: ${JSON.stringify({
      event: 'complete',
      artifact_id: ULID(),
      final_wav: `s3://bucket/final_${session_id}.wav`,
      qc_results
    })}\n\n`);

    res.end();
  } catch (error) {
    if (!res.headersSent) {
      res.status(500).json({ error: String(error) });
    } else {
      res.end();
    }
  }
});

/**
 * Edit Graph 조회
 * GET /api/nde/history/:session_id
 */
ndeRouter.get('/history/:session_id', (req: Request, res: Response) => {
  const { session_id } = req.params;

  const session = sessions.get(session_id);
  if (!session) {
    return res.status(404).json({ error: 'Session not found' });
  }

  const graph = editGraphs.get(session.graph_id);
  if (!graph) {
    return res.status(404).json({ error: 'EditGraph not found' });
  }

  res.json({
    graph_id: graph.graph_id,
    graph_hash: computeGraphHash(graph),
    source_hash: graph.source_hash,
    nodes: graph.nodes,
    created_at: graph.created_at,
    merge_metadata: graph.merge_metadata
  });
});

/**
 * Artifact 다운로드
 * GET /api/nde/artifact/:session_id
 */
ndeRouter.get('/artifact/:session_id', (req: Request, res: Response) => {
  const { session_id } = req.params;

  // Placeholder: 실제로는 S3/Minio에서 artifact 조회
  const artifact = {
    artifact_id: ULID(),
    session_id,
    outputs: {
      final_wav: `s3://bucket/final_${session_id}.wav`,
      preview_mp3: `s3://bucket/preview_${session_id}.mp3`,
      loudness_report: {
        integrated_lufs: -18.5,
        true_peak_db: -2.1,
        lra_lu: 4.2
      }
    },
    qc_results: {
      technical: { passed: true },
      locality: { passed: true },
      similarity: { passed: true }
    }
  };

  res.json(artifact);
});

// ===== Helper Functions =====

async function loadPolicy(policy_id: string): Promise<any> {
  // Placeholder: 실제로는 DB에서 로드
  return {
    policy_id,
    allowed_operations: {
      enhancement_only: true,
      inpaint_allowed: false
    },
    audit_level: 'full_graph',
    qc_gates: {
      technical: { enabled: true },
      locality: { enabled: true },
      similarity: { enabled: false }
    }
  };
}

async function policyGateCheck(node: any, policy: any): Promise<[boolean, string | null]> {
  // Enhancement only 체크
  if (policy.allowed_operations.enhancement_only) {
    const allowed_types = ['dsp', 'export', 'split', 'align', 'mix'];
    if (!allowed_types.includes(node.node_type)) {
      return [false, `Policy violation: enhancement_only mode, got ${node.node_type}`];
    }
  }

  // Inpaint 체크
  if (node.node_type === 'repaint' && !policy.allowed_operations.inpaint_allowed) {
    return [false, 'Policy violation: inpaint not allowed'];
  }

  return [true, null];
}

function computeHash(input: string): string {
  return crypto.createHash('sha256').update(input).digest('hex');
}

function computeGraphHash(graph: any): string {
  const content = JSON.stringify({
    source_hash: graph.source_hash,
    nodes: graph.nodes.map((n: any) => ({
      node_type: n.node_type,
      params: n.params,
      seed: n.seed,
      engine_version: n.engine_version
    }))
  });
  return crypto.createHash('sha256').update(content).digest('hex');
}
