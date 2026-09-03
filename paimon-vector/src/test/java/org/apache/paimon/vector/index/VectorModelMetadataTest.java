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

import org.apache.paimon.globalindex.GlobalIndexSingleColumnWriter;
import org.apache.paimon.globalindex.IvfPqShard;
import org.apache.paimon.globalindex.io.GlobalIndexFileWriter;

import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/** Tests for centroid routing model metadata which do not require the native library. */
class VectorModelMetadataTest {

    private static final String DIGEST =
            "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    @Test
    void testGlobalMetadataRoundTripAndStreamingHeader() throws Exception {
        byte[] payload = "abc".getBytes(StandardCharsets.UTF_8);
        ByteArrayOutputStream artifact = new ByteArrayOutputStream();
        VectorGlobalIndexFileMeta.centroidSharded()
                .writeWithPayload(artifact, new TestTrainingModel(7, payload));

        ByteArrayInputStream in = new ByteArrayInputStream(artifact.toByteArray());
        VectorGlobalIndexFileMeta.ArtifactHeader header =
                VectorGlobalIndexFileMeta.readArtifactHeader(in, artifact.size());

        assertThat(header.metadata().shardMode()).isEqualTo(IvfPqShard.CENTROID_BASED);
        assertThat(header.metadata().nlist()).isEqualTo(7);
        assertThat(header.metadata().modelDigest()).isEqualTo(VectorModelDigest.sha256(payload));
        assertThat(header.payloadSize()).isEqualTo(payload.length);
        assertThat(in.read()).isEqualTo(payload[0]);
    }

    @Test
    void testMetadataRequiresResolvedNlistAndDigest() throws Exception {
        assertThatThrownBy(() -> VectorGlobalIndexFileMeta.centroidSharded().serialize())
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("nlist")
                .hasMessageContaining("modelDigest");
        VectorGlobalIndexFileMeta legacy =
                VectorGlobalIndexFileMeta.deserialize(
                        "{\"version\":1,\"shardMode\":\"centroid-based\"}"
                                .getBytes(StandardCharsets.UTF_8));
        assertThat(legacy.shardMode()).isEqualTo(IvfPqShard.CENTROID_BASED);
        assertThatThrownBy(
                        () ->
                                VectorGlobalIndexFileMeta.deserialize(
                                        "{\"version\":2,\"shardMode\":\"centroid-based\"}"
                                                .getBytes(StandardCharsets.UTF_8)))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("nlist or modelDigest");
    }

    @Test
    void testManifestMetadataCarriesModelIdentity() throws Exception {
        VectorIndexMeta routing =
                VectorIndexMeta.routingModel(IvfPqShard.CENTROID_BASED, 7, DIGEST);
        VectorIndexMeta shard = VectorIndexMeta.centroidShard(3, DIGEST);

        VectorIndexMeta restoredRouting = VectorIndexMeta.deserialize(routing.serialize());
        VectorIndexMeta restoredShard = VectorIndexMeta.deserialize(shard.serialize());

        assertThat(restoredRouting.nlist()).isEqualTo(7);
        assertThat(restoredRouting.modelDigest()).isEqualTo(DIGEST);
        assertThat(restoredShard.centroid()).isEqualTo(3);
        assertThat(restoredShard.modelDigest()).isEqualTo(DIGEST);

        NativeVectorGlobalIndexer indexer =
                new NativeVectorGlobalIndexer(null, Collections.emptyMap(), "ivf-pq");
        assertThat(indexer.isCompatibleRoutedIndexFile(routing.serialize(), shard.serialize()))
                .isTrue();
        assertThat(
                        indexer.isCompatibleRoutedIndexFile(
                                routing.serialize(),
                                VectorIndexMeta.centroidShard(
                                                3,
                                                "sha256:1111111111111111111111111111111111111111111111111111111111111111")
                                        .serialize()))
                .isFalse();
    }

    @Test
    void testDigestIsStableAndValidated() {
        assertThat(VectorModelDigest.sha256("abc".getBytes(StandardCharsets.UTF_8)))
                .isEqualTo(
                        "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assertThatThrownBy(() -> VectorIndexMeta.centroidShard(0, "sha256:ABC"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("64 lowercase hex");
    }

    @Test
    void testLegacyCentroidMetadataRemainsReadableButIsNotRoutable() throws Exception {
        VectorIndexMeta legacy =
                VectorIndexMeta.deserialize(
                        "{\"shardMode\":\"centroid-based\",\"centroid\":\"3\",\"rowIdEncoding\":\"ABSOLUTE_ROW_ID\"}"
                                .getBytes(StandardCharsets.UTF_8));

        assertThat(legacy.isCentroidShard()).isTrue();
        assertThat(legacy.hasModelIdentity()).isFalse();
    }

    @Test
    void testCorruptedPayloadIsRejectedBeforeNativeDeserialization() throws Exception {
        byte[] payload = "abc".getBytes(StandardCharsets.UTF_8);
        byte[] metadata = VectorGlobalIndexFileMeta.centroidSharded(7, DIGEST).serialize();
        ByteArrayOutputStream artifact = new ByteArrayOutputStream();
        try (DataOutputStream out = new DataOutputStream(artifact)) {
            out.writeInt(metadata.length);
            out.write(metadata);
            out.write("changed".getBytes(StandardCharsets.UTF_8));
        }

        assertThatThrownBy(() -> NativeVectorTrainingModels.load(artifact.toByteArray()))
                .isInstanceOf(java.io.IOException.class)
                .hasMessageContaining("digest mismatch");
    }

    private static class TestTrainingModel implements VectorTrainingModel {

        private final int nlist;
        private final byte[] payload;

        private TestTrainingModel(int nlist, byte[] payload) {
            this.nlist = nlist;
            this.payload = payload;
        }

        @Override
        public VectorCentroidModel centroids() {
            return new VectorCentroidModel() {
                @Override
                public int nlist() {
                    return nlist;
                }

                @Override
                public int assignCentroid(float[] vector) {
                    throw new UnsupportedOperationException();
                }

                @Override
                public int[] nearestCentroids(float[] queryVector, int nprobe) {
                    throw new UnsupportedOperationException();
                }
            };
        }

        @Override
        public String modelDigest() {
            return VectorModelDigest.sha256(payload);
        }

        @Override
        public GlobalIndexSingleColumnWriter createCentroidShardIndexWriter(
                GlobalIndexFileWriter fileWriter, int centroid, Map<String, String> options) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void serializeNativeModelPayloadTo(java.io.OutputStream out)
                throws java.io.IOException {
            out.write(payload);
        }

        @Override
        public void close() {}
    }
}
