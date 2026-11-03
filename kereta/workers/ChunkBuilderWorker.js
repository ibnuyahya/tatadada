import { ChunkGeometryCore } from './ChunkGeometryCore.js';

/**
 * ============================================================================
 * BACKGROUND CHUNK BUILDER WEB WORKER (ZERO-COPY TRANSFERABLE BUFFERS)
 * Offloads spline extrusion, 31-column terrain generation, vertex normal math,
 * and multi-species organic vegetation/landmark batching to a background thread.
 * ============================================================================
 */
self.onmessage = function (e) {
  const msg = e.data;
  if (!msg || msg.type !== 'BUILD_CHUNK') return;

  try {
    const data = ChunkGeometryCore.buildChunkBuffers(msg.chunkIdx);
    const transferList = [
      data.ballast.pos.buffer,
      data.ballast.uv.buffer,
      data.ballast.idx.buffer,
      data.ballast.nrm.buffer,
      data.rails.pos.buffer,
      data.rails.col.buffer,
      data.rails.idx.buffer,
      data.rails.nrm.buffer,
      data.wires.pos.buffer,
      data.wires.idx.buffer,
      data.terrain.pos.buffer,
      data.terrain.uv.buffer,
      data.terrain.col.buffer,
      data.terrain.idx.buffer,
      data.terrain.nrm.buffer
    ];
    if (data.scenery.pos.byteLength > 0) {
      transferList.push(
        data.scenery.pos.buffer,
        data.scenery.col.buffer,
        data.scenery.nrm.buffer
      );
    }
    if (data.water && data.water.pos.byteLength > 0) {
      transferList.push(
        data.water.pos.buffer,
        data.water.uv.buffer,
        data.water.col.buffer,
        data.water.idx.buffer,
        data.water.nrm.buffer
      );
    }

    self.postMessage({ type: 'CHUNK_READY', data }, transferList);
  } catch (err) {
    self.postMessage({
      type: 'CHUNK_ERROR',
      chunkIdx: msg.chunkIdx,
      error: err ? String(err.message || err) : 'Unknown worker error'
    });
  }
};
