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

package org.apache.paimon.globalindex;

import javax.annotation.Nullable;

/** Sharding modes for IVF_PQ global indexes. */
public enum IvfPqShard {
    RANGE("range"),
    CENTROID_BASED("centroid-based");

    public static final IvfPqShard DEFAULT = RANGE;

    private final String optionValue;

    IvfPqShard(String optionValue) {
        this.optionValue = optionValue;
    }

    public String optionValue() {
        return optionValue;
    }

    public static IvfPqShard fromOption(@Nullable String value) {
        if (value == null || value.isEmpty()) {
            return DEFAULT;
        }
        for (IvfPqShard shard : values()) {
            if (shard.optionValue.equals(value)) {
                return shard;
            }
        }
        throw new IllegalArgumentException(
                String.format(
                        "Unsupported IVF_PQ shard mode '%s'. Supported values are '%s' and '%s'.",
                        value, RANGE.optionValue, CENTROID_BASED.optionValue));
    }

    @Nullable
    public static IvfPqShard fromValue(@Nullable String value) {
        if (value == null || value.isEmpty()) {
            return null;
        }
        for (IvfPqShard shard : values()) {
            if (shard.optionValue.equals(value) || shard.name().equals(value)) {
                return shard;
            }
        }
        if ("centroidId".equals(value) || "CENTROID_ID".equals(value)) {
            return CENTROID_BASED;
        }
        return null;
    }
}
