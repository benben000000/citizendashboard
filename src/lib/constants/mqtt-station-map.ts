/**
 * Station identity bridge between the public dashboard station IDs and the
 * AWS IoT device IDs used by the MQTT ingestor.
 *
 * WHY THIS EXISTS
 * ---------------
 * The Python ingestor (`prediction-model/src/mqtt_live_ingestor.py`) subscribes
 * to `kloudtrack/+/data` and keys its cache by the AWS IoT device id
 * (`KT-XXXXXXXXXXXX`). The public dashboard identifies stations by Kloudtrack
 * public ids (`O3z0j5bG`, `95pM7BAV`, ...). These are DIFFERENT namespaces for
 * the same physical hardware.
 *
 * Before this map existed, `telemetry.service.ts` guessed with
 * `mqttData[sid] || mqttData[`KT-${sid}`] || ...`, which matches almost nothing:
 * only 2 of the 16 stations in the live MQTT cache had an id that appeared in
 * the dashboard station list. Every other station silently fell through to
 * hardcoded climatology defaults.
 *
 * SOURCE OF TRUTH
 * ---------------
 * `mqtt/mqtt-needs.txt` is the device-provisioning list and is the authoritative
 * human-name -> device-id mapping. It is reproduced here as typed data because
 * the web app cannot read from the `mqtt/` credential directory at runtime.
 *
 * When a station cannot be resolved, callers MUST surface that fact. Silently
 * substituting a different station's forecast is worse than showing none.
 */

export interface MqttDeviceStation {
  /** Canonical device label from the provisioning list. */
  readonly deviceLabel: string;
  /** AWS IoT device id / MQTT topic segment. */
  readonly deviceId: string;
  /** Normalised key used for fuzzy matching against dashboard station names. */
  readonly matchKey: string;
}

/** Normalise a station label for robust cross-namespace matching. */
export function normaliseStationName(value: string): string {
  return value
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "") // strip diacritics (Doña -> Dona)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

/**
 * Device provisioning list, transcribed from `mqtt/mqtt-needs.txt`.
 * Keys are normalised device labels.
 */
const DEVICE_STATIONS: readonly MqttDeviceStation[] = [
  { deviceLabel: "Old Cabcaben Pier - Bataan", deviceId: "KT-6CBD47DC5194", matchKey: "old cabcaben pier bataan" },
  { deviceLabel: "Dinalupihan - Bataan", deviceId: "KT-CC380371FE68", matchKey: "dinalupihan bataan" },
  { deviceLabel: "Dona Maria", deviceId: "KT-B850AD182EC8", matchKey: "dona maria" },
  { deviceLabel: "Pag Asa Orani", deviceId: "KT-A86039DC5194", matchKey: "pag asa orani" },
  { deviceLabel: "1Bataan Command Center", deviceId: "KT-8CEE47DC5194", matchKey: "1bataan command center" },
  { deviceLabel: "General Natividad", deviceId: "KT-E0B89EF7A608", matchKey: "general natividad" },
  { deviceLabel: "Calumpit WLMS", deviceId: "KT-4049D3215788", matchKey: "calumpit wlms" },
  { deviceLabel: "Calumpit AWS", deviceId: "KT-4C31325C7BCC", matchKey: "calumpit aws" },
  { deviceLabel: "Bongabon", deviceId: "KT-245EAD182EC8", matchKey: "bongabon" },
  { deviceLabel: "Pag-Asa Bagac", deviceId: "KT-3CCCAC182EC8", matchKey: "pag asa bagac" },
  { deviceLabel: "Poblacion Mariveles", deviceId: "KT-D032325C7BCC", matchKey: "poblacion mariveles" },
  { deviceLabel: "Abucay AWS", deviceId: "KT-D831325C7BCC", matchKey: "abucay aws" },
  { deviceLabel: "Avida Asten AWS", deviceId: "KT-A80A1B29E748", matchKey: "avida asten aws" },
  { deviceLabel: "San Jose City", deviceId: "KT-B82DB21C0610", matchKey: "san jose city" },
  { deviceLabel: "San Luis AWS", deviceId: "KT-5C74AC182EC8", matchKey: "san luis aws" },
  { deviceLabel: "Lazatin AWS", deviceId: "KT-20FCA4182EC8", matchKey: "lazatin aws" },
  { deviceLabel: "Baretto AWS", deviceId: "KT-184AAD182EC8", matchKey: "baretto aws" },
  { deviceLabel: "Old Cabalan AWS", deviceId: "KT-EC4FAD182EC8", matchKey: "old cabalan aws" },
  { deviceLabel: "Sabang Morong AWS", deviceId: "KT-183017F7A608", matchKey: "sabang morong aws" },
  { deviceLabel: "Wawa Limay AWS", deviceId: "KT-94AD8332A7B0", matchKey: "wawa limay aws" },
  { deviceLabel: "Alasas AWS", deviceId: "KT-BC25B61815AC", matchKey: "alasas aws" },
  { deviceLabel: "Sapang Buho AWS", deviceId: "KT-3C50AD182EC8", matchKey: "sapang buho aws" },
  { deviceLabel: "Popolon AWS", deviceId: "KT-8050AD182EC8", matchKey: "popolon aws" },
  { deviceLabel: "Hermosa", deviceId: "KT-E01ED2215788", matchKey: "hermosa" },
] as const;

const BY_DEVICE_ID: ReadonlyMap<string, MqttDeviceStation> = new Map(
  DEVICE_STATIONS.map((s) => [s.deviceId, s])
);

const BY_MATCH_KEY: ReadonlyMap<string, MqttDeviceStation> = new Map(
  DEVICE_STATIONS.map((s) => [s.matchKey, s])
);

/** All provisionable devices, for diagnostics and coverage reporting. */
export const ALL_MQTT_DEVICE_STATIONS: readonly MqttDeviceStation[] = DEVICE_STATIONS;

export function getDeviceStationById(deviceId: string): MqttDeviceStation | undefined {
  if (!deviceId) return undefined;
  return BY_DEVICE_ID.get(deviceId) ?? BY_DEVICE_ID.get(`KT-${deviceId}`);
}

/** Levenshtein distance, bounded for short station labels. */
function editDistance(a: string, b: string): number {
  if (a === b) return 0;
  if (a.length === 0) return b.length;
  if (b.length === 0) return a.length;
  let prev = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    const curr = [i];
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      curr[j] = Math.min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost);
    }
    prev = curr;
  }
  return prev[b.length];
}

/**
 * Resolve a dashboard station (public id + display name) to its MQTT device id.
 *
 * Resolution order:
 *   1. The station id is already a device id (`KT-...`).
 *   2. The station name matches a provisioned device label exactly (normalised).
 *   3. The leading segment of a "<label> - <place>" name matches a label.
 *   4. Containment match, longest label first.
 *   5. Spelling-tolerant match. The provisioning list and the dashboard disagree
 *      on at least one label ("Baretto" vs "Barretto"), which previously left
 *      that station permanently unresolved even while its device was reporting
 *      live telemetry.
 *
 * Returns `null` when the station is not in the MQTT fleet. Callers must treat
 * that as "no model forecast available" rather than falling back to synthetic
 * values.
 */
export function resolveDeviceId(
  stationPublicId: string | null | undefined,
  stationName: string | null | undefined
): string | null {
  if (stationPublicId && BY_DEVICE_ID.has(stationPublicId)) {
    return stationPublicId;
  }

  if (stationName) {
    const key = normaliseStationName(stationName);
    if (!key) return null;

    // Exact normalised match.
    const exact = BY_MATCH_KEY.get(key);
    if (exact) return exact.deviceId;

    // Dashboard names are usually "<Device Label> - <City/Province>"; try the
    // leading segment before the first hyphen.
    const [head] = key.split(" - ");
    if (head && head !== key) {
      const byHead = BY_MATCH_KEY.get(head);
      if (byHead) return byHead.deviceId;
    }

    // Containment match, longest label first to avoid a short label shadowing
    // a more specific one ("pag asa bagac" vs "pag asa orani").
    const contained = [...BY_MATCH_KEY.entries()]
      .filter(([label]) => key.includes(label) || label.includes(key))
      .sort((a, b) => b[0].length - a[0].length);
    if (contained.length > 0) {
      return contained[0][1].deviceId;
    }

    // Spelling-tolerant fallback.
    //
    // The provisioning list and the dashboard disagree on at least one label
    // ("Baretto" vs "Barretto"), which previously left that station permanently
    // unresolved even while its device reported live telemetry.
    //
    // Scoring is on the single best DISTINCTIVE token match rather than a whole
    // name comparison, because dashboard names append a place ("- Olongapo
    // City") that has no counterpart in the provisioning label and would
    // otherwise dominate the score. A winner must be strictly better than any
    // runner-up so we never guess between two stations.
    const queryTokens = key.split(" ").filter((t) => t.length >= 5);
    const scores: Array<{ deviceId: string; label: string; score: number }> = [];
    for (const [label, station] of BY_MATCH_KEY) {
      const labelTokens = label.split(" ").filter((t) => t.length >= 5);
      if (queryTokens.length === 0 || labelTokens.length === 0) continue;
      let best = Number.POSITIVE_INFINITY;
      for (const qt of queryTokens) {
        for (const lt of labelTokens) {
          const d = editDistance(qt, lt);
          if (d < best) best = d;
        }
      }
      // One typo for short labels, up to two for long ones.
      const longest = Math.max(...labelTokens.map((t) => t.length));
      const budget = longest >= 8 ? 2 : 1;
      if (best <= budget) {
        scores.push({ deviceId: station.deviceId, label, score: best });
      }
    }
    scores.sort((a, b) => a.score - b.score);
    if (scores.length > 0 && (scores.length === 1 || scores[0].score < scores[1].score)) {
      return scores[0].deviceId;
    }
  }

  return null;
}
