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

package org.apache.paimon.spark.globalindex;

import org.apache.paimon.spark.globalindex.SparkVectorTrainingOrchestrator.Configuration;
import org.apache.paimon.spark.globalindex.SparkVectorTrainingOrchestrator.TrainingResult;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/** Tests for {@link NativeCoarseTrainingEngine}. */
public class NativeCoarseTrainingEngineTest {

    @Test
    void testNativeCoarseTrainingRoundTrip() throws Exception {
        String nativeLibrary = System.getProperty("paimon.vindex.native.path");
        Assumptions.assumeTrue(
                nativeLibrary != null
                        && !nativeLibrary.trim().isEmpty()
                        && Files.isRegularFile(Paths.get(nativeLibrary)),
                "Native distributed training library is not configured.");
        NativeCoarseTrainingEngine engine =
                new NativeCoarseTrainingEngine(
                        2, 2, 8, 0.0d, new float[] {0.0f, 0.0f, 10.0f, 10.0f});

        TrainingResult result =
                SparkVectorTrainingOrchestrator.trainLocal(
                        Arrays.asList(
                                Arrays.asList(vector(0.0f, 0.0f), vector(0.2f, 0.1f)),
                                Collections.singletonList(vector(9.8f, 10.1f)),
                                Collections.singletonList(vector(10.0f, 10.0f))),
                        engine,
                        new Configuration(8, 1024 * 1024, 8L * 1024 * 1024));

        assertThat(result.model()).isNotEmpty();
        assertThat(result.finalState()).isNotEmpty();
        assertThat(result.iterations()).isBetween(1, 8);
        assertThat(result.vectorCount()).isEqualTo(4);
        assertThat(result.partitionCount()).isEqualTo(3);
    }

    @Test
    void testRejectsInvalidConfigurationBeforeNativeCall() {
        assertThatThrownBy(
                        () ->
                                new NativeCoarseTrainingEngine(
                                        2, 2, 8, Double.NaN, new float[] {0, 0, 1, 1}))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("tolerance");

        assertThatThrownBy(
                        () -> new NativeCoarseTrainingEngine(2, 2, 8, 0.0d, new float[] {0, 0, 1}))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("dimension * nlist");
    }

    private static float[] vector(float x, float y) {
        return new float[] {x, y};
    }
}
