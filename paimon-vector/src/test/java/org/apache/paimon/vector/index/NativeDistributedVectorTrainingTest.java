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

import org.apache.paimon.index.vector.VectorIndexTrainer;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/** Tests for {@link NativeDistributedVectorTraining}. */
class NativeDistributedVectorTrainingTest {

    @Test
    void testFinalTrainingOptionsAndRawSampleAreValidatedBeforeNativeCall() {
        Map<String, String> options = options();

        assertThatThrownBy(
                        () ->
                                NativeDistributedVectorTraining.trainIvfPq(
                                        "ivf-flat",
                                        options,
                                        new byte[] {1},
                                        new float[] {0.0f, 0.0f},
                                        1))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("only index type 'ivf-pq'");

        options.put("metric", "cosine");
        assertThatThrownBy(
                        () ->
                                NativeDistributedVectorTraining.trainIvfPq(
                                        "ivf-pq",
                                        options,
                                        new byte[] {1},
                                        new float[] {0.0f, 0.0f},
                                        1))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("metric=l2");

        options.put("metric", "l2");
        options.put("use-opq", "true");
        assertThatThrownBy(
                        () ->
                                NativeDistributedVectorTraining.trainIvfPq(
                                        "ivf-pq",
                                        options,
                                        new byte[] {1},
                                        new float[] {0.0f, 0.0f},
                                        1))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("use-opq=false");

        options.put("use-opq", "false");
        options.put("nlist", "auto");
        assertThatThrownBy(
                        () ->
                                NativeDistributedVectorTraining.trainIvfPq(
                                        "ivf-pq",
                                        options,
                                        new byte[] {1},
                                        new float[] {0.0f, 0.0f},
                                        1))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("fixed positive option 'nlist'");

        options.put("nlist", "1");
        assertThatThrownBy(
                        () ->
                                NativeDistributedVectorTraining.trainIvfPq(
                                        "ivf-pq", options, new byte[] {1}, new float[] {0.0f}, 1))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("rawSample length");
    }

    @Test
    void testCoarseLifecycleAndFinalModel() throws Exception {
        Assumptions.assumeTrue(isNativeAvailable(), "Vector index native library not available");

        byte[] state =
                NativeDistributedVectorTraining.initializeCoarse(
                        2, 1, 1, 0.0d, new float[] {0.0f, 0.0f});
        byte[] partial;
        try (NativeDistributedVectorTraining.CoarseAccumulator accumulator =
                NativeDistributedVectorTraining.createCoarseAccumulator(state, 0L)) {
            accumulator.addBatch(new float[] {0.0f, 0.0f, 2.0f, 0.0f}, 2);
            partial = accumulator.finish();
        }
        byte[] aggregate =
                NativeDistributedVectorTraining.mergeCoarse(state, Arrays.asList(partial));
        NativeDistributedVectorTraining.CoarseIteration iteration =
                NativeDistributedVectorTraining.advanceCoarse(state, aggregate);

        assertThat(iteration.finished()).isTrue();
        assertThat(iteration.iteration()).isEqualTo(1);
        assertThat(javaSerializationRoundTrip(iteration).state())
                .containsExactly(iteration.state());

        byte[] coarseModel = NativeDistributedVectorTraining.finalizeCoarse(iteration.state());
        float[] rawSample = new float[32 * 2];
        for (int row = 0; row < 32; row++) {
            rawSample[row * 2] = row * 0.01f;
            rawSample[row * 2 + 1] = row * 0.02f;
        }
        try (VectorTrainingModel model =
                NativeDistributedVectorTraining.trainIvfPq(
                        "ivf-pq", options(), coarseModel, rawSample, 32)) {
            assertThat(model.centroids().nlist()).isEqualTo(1);
            ByteArrayOutputStream serialized = new ByteArrayOutputStream();
            model.serializeNativeModelPayloadTo(serialized);
            assertThat(serialized.size()).isPositive();
        }
    }

    private static Map<String, String> options() {
        Map<String, String> options = new LinkedHashMap<>();
        options.put("index.type", "ivf_pq");
        options.put("dimension", "2");
        options.put("nlist", "1");
        options.put("metric", "l2");
        options.put("pq.m", "1");
        options.put("use-opq", "false");
        return options;
    }

    private static boolean isNativeAvailable() {
        try (VectorIndexTrainer ignored = VectorIndexTrainer.create(options())) {
            return true;
        } catch (Throwable t) {
            return false;
        }
    }

    @SuppressWarnings("unchecked")
    private static <T> T javaSerializationRoundTrip(T value) throws Exception {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (ObjectOutputStream output = new ObjectOutputStream(bytes)) {
            output.writeObject(value);
        }
        try (ObjectInputStream input =
                new ObjectInputStream(new ByteArrayInputStream(bytes.toByteArray()))) {
            return (T) input.readObject();
        }
    }
}
