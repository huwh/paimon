/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.paimon.vector.index;

import org.apache.paimon.fs.PositionOutputStream;
import org.apache.paimon.globalindex.GlobalIndexSingleColumnWriter;
import org.apache.paimon.globalindex.ResultEntry;
import org.apache.paimon.globalindex.io.GlobalIndexFileWriter;
import org.apache.paimon.index.vector.VectorIndexWriter;
import org.apache.paimon.utils.Range;

import javax.annotation.Nullable;

import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/** Writes one physical vector index file for one centroid. */
public class NativeCentroidShardIndexWriter implements GlobalIndexSingleColumnWriter, Closeable {

    // ':' is parsed as a URI scheme separator by Path and is not valid in a relative file name.
    private static final String FILE_NAME_PREFIX = "vector-centroid";
    private static final int IO_BUFFER_SIZE = 8 * 1024 * 1024;
    private static final int ADD_BATCH_SIZE = 10000;

    private final GlobalIndexFileWriter fileWriter;
    private final NativeVectorGlobalModelTrainers.NativeVectorTrainingModel trainingModel;
    private final int centroid;
    private final int dim;
    private final int recordSizeInBytes;

    private File tempVectorFile;
    private FileChannel writeChannel;
    private ByteBuffer writeBuf;
    private long count;
    private long minRowId = Long.MAX_VALUE;
    private long maxRowId = Long.MIN_VALUE;
    private boolean finished;

    public NativeCentroidShardIndexWriter(
            GlobalIndexFileWriter fileWriter,
            NativeVectorGlobalModelTrainers.NativeVectorTrainingModel trainingModel,
            int centroid,
            Map<String, String> nativeOptions) {
        this.fileWriter = fileWriter;
        this.trainingModel = trainingModel;
        this.centroid = centroid;
        this.dim = parseDimension(nativeOptions);
        this.recordSizeInBytes =
                NativeVectorGlobalIndexWriter.checkedRecordSize(dim, IO_BUFFER_SIZE);
        try {
            this.tempVectorFile = File.createTempFile("paimon-vector-centroid-shard-", ".bin");
            this.tempVectorFile.deleteOnExit();
            @SuppressWarnings("resource")
            RandomAccessFile raf = new RandomAccessFile(tempVectorFile, "rw");
            this.writeChannel = raf.getChannel();
            this.writeBuf = ByteBuffer.allocateDirect(IO_BUFFER_SIZE);
            this.writeBuf.order(ByteOrder.nativeOrder());
        } catch (IOException e) {
            throw new RuntimeException("Failed to create temp vector file for centroid shard", e);
        }
    }

    @Override
    public void write(@Nullable Object vector, long absoluteRowId) {
        ensureNotFinished();
        if (vector == null) {
            return;
        }
        float[] materialized = materializeAndValidate(vector, absoluteRowId);
        if (writeBuf.remaining() < recordSizeInBytes) {
            flushWriteBuffer();
        }
        writeBuf.putLong(absoluteRowId);
        for (int i = 0; i < dim; i++) {
            writeBuf.putFloat(materialized[i]);
        }
        count++;
        minRowId = Math.min(minRowId, absoluteRowId);
        maxRowId = Math.max(maxRowId, absoluteRowId);
    }

    @Override
    public List<ResultEntry> finish() {
        ensureNotFinished();
        finished = true;
        Throwable failure = null;
        try {
            if (count == 0) {
                return Collections.emptyList();
            }
            flushWriteBuffer();
            closeWriteChannel();

            String fileName = fileWriter.newFileName(FILE_NAME_PREFIX + "-" + centroid);
            try (VectorIndexWriter writer =
                    trainingModel.createIvfPqCentroidShardWriter(centroid)) {
                addVectorsFromTempFile(writer);
                try (PositionOutputStream out = fileWriter.newOutputStream(fileName)) {
                    writer.writeIndex(out);
                    out.flush();
                }
            }
            return Collections.singletonList(
                    new ResultEntry(
                            fileName,
                            count,
                            VectorIndexMeta.centroidShard(centroid, trainingModel.modelDigest())
                                    .serialize(),
                            new Range(minRowId, maxRowId)));
        } catch (IOException e) {
            RuntimeException wrapped =
                    new RuntimeException(
                            "Failed to write native centroid shard index for centroid=" + centroid,
                            e);
            failure = wrapped;
            throw wrapped;
        } catch (RuntimeException | Error e) {
            failure = e;
            throw e;
        } finally {
            try {
                close();
            } catch (RuntimeException closeFailure) {
                if (failure == null) {
                    throw closeFailure;
                }
                failure.addSuppressed(closeFailure);
            }
        }
    }

    @Override
    public void close() {
        try {
            closeWriteChannel();
        } finally {
            writeBuf = null;
            if (tempVectorFile != null) {
                tempVectorFile.delete();
                tempVectorFile = null;
            }
        }
    }

    private void flushWriteBuffer() {
        try {
            writeBuf.flip();
            while (writeBuf.hasRemaining()) {
                writeChannel.write(writeBuf);
            }
            writeBuf.clear();
        } catch (IOException e) {
            throw new RuntimeException("Failed to flush centroid shard vector buffer to disk", e);
        }
    }

    private void closeWriteChannel() {
        if (writeChannel != null) {
            try {
                writeChannel.close();
            } catch (IOException e) {
                throw new RuntimeException("Failed to close centroid shard temp vector file", e);
            } finally {
                writeChannel = null;
            }
        }
    }

    private void addVectorsFromTempFile(VectorIndexWriter writer) throws IOException {
        int addBatchSize = NativeVectorGlobalIndexWriter.vectorBatchSize(ADD_BATCH_SIZE, dim);
        long[] batchIds = new long[addBatchSize];
        float[] batchVectors = new float[addBatchSize * dim];

        try (RandomAccessFile raf = new RandomAccessFile(tempVectorFile, "r");
                FileChannel channel = raf.getChannel()) {
            ByteBuffer readBuf = ByteBuffer.allocateDirect(IO_BUFFER_SIZE);
            readBuf.order(ByteOrder.nativeOrder());
            readBuf.limit(0);

            long remaining = count;
            while (remaining > 0) {
                int thisBatch = (int) Math.min(addBatchSize, remaining);
                for (int i = 0; i < thisBatch; i++) {
                    ensureAvailable(readBuf, channel, recordSizeInBytes);
                    batchIds[i] = readBuf.getLong();
                    for (int d = 0; d < dim; d++) {
                        batchVectors[i * dim + d] = readBuf.getFloat();
                    }
                }
                if (thisBatch == addBatchSize) {
                    trainingModel.addIvfPqCentroidVectors(
                            writer, batchIds, batchVectors, thisBatch);
                } else {
                    trainingModel.addIvfPqCentroidVectors(
                            writer,
                            Arrays.copyOf(batchIds, thisBatch),
                            Arrays.copyOf(batchVectors, thisBatch * dim),
                            thisBatch);
                }
                remaining -= thisBatch;
            }
        }
    }

    private static void ensureAvailable(ByteBuffer readBuf, FileChannel channel, int minBytes)
            throws IOException {
        int zeroReadCount = 0;
        while (readBuf.remaining() < minBytes) {
            readBuf.compact();
            int bytesRead = channel.read(readBuf);
            readBuf.flip();
            if (bytesRead == -1) {
                throw new IOException("Unexpected end of centroid shard temp file");
            }
            if (bytesRead == 0) {
                if (++zeroReadCount > 100) {
                    throw new IOException(
                            "Unable to read from centroid shard temp file: repeated zero-byte reads");
                }
            } else {
                zeroReadCount = 0;
            }
        }
    }

    private float[] materializeAndValidate(Object vector, long absoluteRowId) {
        if (vector instanceof float[]) {
            float[] floats = (float[]) vector;
            checkDimension(floats.length);
            for (int i = 0; i < dim; i++) {
                checkFinite(floats[i], absoluteRowId, i);
            }
            return floats;
        }
        throw new IllegalArgumentException(
                "Centroid shard writer expects float[] vector assigned by centroid builder, but got: "
                        + vector.getClass().getName());
    }

    private void checkDimension(int actualDim) {
        if (actualDim != dim) {
            throw new IllegalArgumentException(
                    String.format(
                            "Vector dimension mismatch: expected %d, but got %d", dim, actualDim));
        }
    }

    private void checkFinite(float value, long absoluteRowId, int elementIndex) {
        if (!Float.isFinite(value)) {
            throw new IllegalArgumentException(
                    String.format(
                            "Vector element at rowId=%d, index=%d is %s",
                            absoluteRowId, elementIndex, Float.toString(value)));
        }
    }

    private void ensureNotFinished() {
        if (finished) {
            throw new IllegalStateException(
                    "Native centroid shard writer has already finished. centroid=" + centroid);
        }
    }

    private static int parseDimension(Map<String, String> nativeOptions) {
        String dimension = nativeOptions.get("dimension");
        if (dimension == null) {
            throw new IllegalArgumentException(
                    "Native centroid shard writer requires option 'dimension'.");
        }
        int dim = Integer.parseInt(dimension);
        if (dim <= 0) {
            throw new IllegalArgumentException(
                    "Native centroid shard writer requires positive dimension.");
        }
        return dim;
    }
}
