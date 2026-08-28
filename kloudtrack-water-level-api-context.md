# KloudTrack Water Level API Context

This document is intended to give an AI enough context to understand and use the KloudTrack **water-level** endpoints.

## Base API Information

**Base URL**

```txt
https://api.kloudtechsea.com/api/v1
```

**Authentication**

All endpoints require the `x-kloudtrack-key` header.

```http
x-kloudtrack-key: YOUR_API_KEY_HERE
```

**Standard response envelope**

Most responses follow this shape:

```ts
type ApiResponse<T> = {
  success: boolean;
  message: string;
  data: T | null;
};
```

**Access control**

- API keys are scoped to the authenticated user's account and organization.
- Users can only access stations within their own organization or stations they have permission to read.
- Water-level endpoints require read access to the station/water-level data.

**Rate limit**

```txt
20 requests per minute per account
```

If exceeded, the API may return `429 Too Many Requests`.

---

## Core Concepts

### StationInfo

Water-level responses usually include parent station metadata.

```ts
type StationInfo = {
  id: string; // station hashid, e.g. "st_abc123"
  stationName: string;
  stationType: string; // often "RIVERLEVEL" or a modular station with water-level module
  location?: {
    lat: number;
    lng: number;
  };
  address?: string;
  city?: string;
  state?: string;
  country?: string;
  elevation?: number;
  isActive?: boolean;
  activatedAt?: string; // ISO 8601 timestamp
  organizationId?: number;
  organization?: {
    id: number;
    organizationName: string;
  };
};
```

### WaterLevelReading

Water-level readings are aggregate sensor rows. The most important fields are usually `recordedAt`, `rawMode`, and `calculatedWaterLevel`.

```ts
type WaterLevelReading = {
  id: number;
  recordedAt: string; // ISO 8601 timestamp

  startTimestamp?: string | null;
  endTimestamp?: string | null;

  sampleInterval?: number | null;
  sampleCount?: number | null;
  filteredSampleCount?: number | null;
  spikeCount?: number | null;

  minimum?: number | null;
  maximum?: number | null;
  rawMode?: number | null;
  calculatedWaterLevel?: number | null;
  median?: number | null;
  frequentRangeLow?: number | null;
  frequentRangeHigh?: number | null;
  estimatedMovAvg?: number | null;
};
```

### Meaning of `calculatedWaterLevel`

`calculatedWaterLevel` is derived from the raw sensor value, commonly `rawMode`, and the per-station reference configuration when both are present.

Treat units as deployment-specific. In many deployments, it represents centimeters above a configured reference point.

`calculatedWaterLevel` may be `null` when the station reference configuration or raw reading data is missing.

### VariableReading

Single-variable history endpoints return compact time-series points.

```ts
type VariableReading = {
  id: number;
  recordedAt: string;
  createdAt: string;
  value: number | null;
};
```

---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     

## Endpoint Summary

| Purpose | Method | Path |
|---|---:|---|
| Fleet/current dashboard | GET | `/water-level/dashboard` |
| Fetch one water-level record by ID | GET | `/water-level/record/{id}` |
| Latest reading for one station | GET | `/water-level/station/{stationId}/current` |
| Full water-level history for one station | GET | `/water-level/station/{stationId}/history` |
| Single-variable water-level history | GET | `/water-level/station/{stationId}/history/{variable}` |

---

## 1. Get Water Level Dashboard Stations

```http
GET /water-level/dashboard
```

Use this to fetch all readable water-level stations for the authenticated account, each paired with its latest water-level reading.

This is the best endpoint for overview maps, dashboard cards, alert tiles, and fleet-level current status views.

### Response data shape

```ts
type WaterLevelDashboardResponse = {
  station: StationInfo;
  waterLevel: WaterLevelReading | null;
}[];
```

`waterLevel` can be `null` when the station is eligible for water-level data but has never ingested a reading.

### Example request

```ts
const response = await fetch(
  'https://api.kloudtechsea.com/api/v1/water-level/dashboard',
  {
    headers: {
      'x-kloudtrack-key': 'YOUR_API_KEY_HERE',
    },
  }
);

const json = await response.json();

if (!response.ok) {
  throw new Error(JSON.stringify(json));
}

console.log(json.data);
```

### Example response

```json
{
  "success": true,
  "message": "Water level dashboard data retrieved successfully",
  "data": [
    {
      "station": {
        "id": "st_abc123",
        "stationName": "River gauge 01",
        "stationType": "RIVERLEVEL",
        "location": { "lat": 23.5, "lng": 120.2 },
        "address": "1 River Rd",
        "city": "Taichung",
        "state": "",
        "country": "TW",
        "elevation": 42,
        "isActive": true,
        "activatedAt": "2026-01-10T00:00:00.000Z",
        "organizationId": 10,
        "organization": {
          "id": 10,
          "organizationName": "Demo Org"
        }
      },
      "waterLevel": {
        "id": 5001,
        "recordedAt": "2026-05-04T08:00:00.000Z",
        "startTimestamp": null,
        "endTimestamp": null,
        "sampleInterval": 60,
        "sampleCount": 120,
        "filteredSampleCount": 118,
        "spikeCount": 2,
        "minimum": 95.2,
        "maximum": 98.1,
        "rawMode": 96.4,
        "calculatedWaterLevel": 101.2,
        "median": 96.5,
        "frequentRangeLow": 95.8,
        "frequentRangeHigh": 97.2,
        "estimatedMovAvg": 96.6
      }
    }
  ]
}
```

### Notes for AI usage

- Use this instead of making many `/current` calls when showing many stations.
- Always handle `waterLevel: null`.
- Station IDs are string hashids like `st_abc123`.

---

## 2. Get Water Level Record by ID

```http
GET /water-level/record/{id}
```

Fetches one stored water-level aggregate row by numeric water-level reading ID.

Use this when an app already has a `waterLevel.id` from dashboard, history, webhooks, exports, logs, or saved references.

### Path parameters

| Name | Type | Required | Description |
|---|---|---:|---|
| `id` | number | Yes | Numeric water-level reading ID, e.g. `5001`. This is **not** the station ID. |

### Response data shape

```ts
type WaterLevelRecordResponse = {
  station: StationInfo;
  waterLevel: WaterLevelReading;
};
```

### Example request

```ts
const id = 5001;

const response = await fetch(
  `https://api.kloudtechsea.com/api/v1/water-level/record/${id}`,
  {
    headers: {
      'x-kloudtrack-key': 'YOUR_API_KEY_HERE',
    },
  }
);

const json = await response.json();
```

### Example response

```json
{
  "success": true,
  "message": "Water level data retrieved successfully",
  "data": {
    "station": {
      "id": "st_abc123",
      "stationName": "River gauge 01",
      "stationType": "RIVERLEVEL"
    },
    "waterLevel": {
      "id": 5001,
      "recordedAt": "2026-05-04T08:00:00.000Z",
      "calculatedWaterLevel": 101.2,
      "rawMode": 96.4,
      "median": 96.5,
      "minimum": 95.2,
      "maximum": 98.1,
      "sampleInterval": 60,
      "sampleCount": 120,
      "filteredSampleCount": 118,
      "spikeCount": 2,
      "frequentRangeLow": 95.8,
      "frequentRangeHigh": 97.2,
      "estimatedMovAvg": 96.6,
      "startTimestamp": null,
      "endTimestamp": null
    }
  }
}
```

### Common errors

| Status | Meaning |
|---:|---|
| 404 | No water-level row exists for that ID. |
| 403 | The row exists but belongs to a station the authenticated account cannot read. |

---

## 3. Get Latest Water Level Reading for a Station

```http
GET /water-level/station/{stationId}/current
```

Returns the most recent water-level aggregate reading for a specific station.

Use this for single-station widgets, station detail pages, or alert checks when the station ID is already known.

### Path parameters

| Name | Type | Required | Description |
|---|---|---:|---|
| `stationId` | string | Yes | Station hashid such as `st_abc123`, or an encoded numeric form accepted by the server. |

### Response data shape

```ts
type CurrentWaterLevelResponse = {
  station: StationInfo;
  waterLevel: WaterLevelReading;
};
```

### Example request

```ts
const stationId = 'st_abc123';

const response = await fetch(
  `https://api.kloudtechsea.com/api/v1/water-level/station/${stationId}/current`,
  {
    headers: {
      'x-kloudtrack-key': 'YOUR_API_KEY_HERE',
    },
  }
);

const json = await response.json();
```

### Example response

```json
{
  "success": true,
  "message": "Current water level data retrieved successfully",
  "data": {
    "station": {
      "id": "st_abc123",
      "stationName": "River gauge 01",
      "stationType": "RIVERLEVEL"
    },
    "waterLevel": {
      "id": 5001,
      "recordedAt": "2026-05-04T08:00:00.000Z",
      "calculatedWaterLevel": 101.2,
      "rawMode": 96.4
    }
  }
}
```

### Common errors

| Status | Meaning |
|---:|---|
| 400 | Invalid station ID encoding. |
| 404 | The station has no water-level rows yet. |

### Notes for AI usage

- Latest is determined server-side using the most recent reading ordering.
- For multiple stations, prefer `/water-level/dashboard` over calling this repeatedly.

---

## 4. Get Water Level History for a Station

```http
GET /water-level/station/{stationId}/history
```

Returns a time series of water-level aggregate rows for one station.

Use this for charts, tables, CSV exports, analytics, and historical review.

Records are returned newest first. Reverse the array client-side when plotting charts that expect ascending time order.

### Path parameters

| Name | Type | Required | Description |
|---|---|---:|---|
| `stationId` | string | Yes | Station hashid for the water-level station. |

### Query parameters

| Name | Type | Required | Description |
|---|---|---:|---|
| `skip` | number | No | SQL offset for pagination. Defaults to `0`. |
| `take` | number | No | Page size. Use with date windows for large histories. |
| `interval` | number | No | Aggregation bucket size in minutes. Allowed: `1`, `15`, `30`, `60`, `180`, `360`, `720`, `1440`. Omit for raw per-row data. |
| `startDate` | string | No | Inclusive lower bound on `recordedAt`, ISO 8601 format. Recommended for non-trivial queries. |
| `endDate` | string | No | Inclusive upper bound on `recordedAt`, ISO 8601 format. Defaults to now when omitted. |

### Response data shape

```ts
type WaterLevelHistoryResponse = {
  station: StationInfo;
  waterLevel: WaterLevelReading[];
};
```

### Example request

```ts
const stationId = 'st_abc123';

const params = new URLSearchParams({
  skip: '0',
  interval: '60',
  startDate: '2026-05-01T00:00:00.000Z',
  endDate: '2026-05-04T23:59:59.999Z',
});

const response = await fetch(
  `https://api.kloudtechsea.com/api/v1/water-level/station/${stationId}/history?${params}`,
  {
    headers: {
      'x-kloudtrack-key': 'YOUR_API_KEY_HERE',
    },
  }
);

const json = await response.json();

// For charts, reverse from newest-first to oldest-first.
const chartData = [...json.data.waterLevel].reverse();
```

### Example response

```json
{
  "success": true,
  "message": "Water level history retrieved successfully",
  "data": {
    "station": {
      "id": "st_abc123",
      "stationName": "River gauge 01",
      "stationType": "RIVERLEVEL"
    },
    "waterLevel": [
      {
        "id": 5002,
        "recordedAt": "2026-05-04T09:00:00.000Z",
        "calculatedWaterLevel": 101.5,
        "rawMode": 96.6,
        "median": 96.7,
        "minimum": 95.4,
        "maximum": 98.0,
        "sampleInterval": 60,
        "sampleCount": 120,
        "filteredSampleCount": 118,
        "spikeCount": 2,
        "frequentRangeLow": 95.9,
        "frequentRangeHigh": 97.3,
        "estimatedMovAvg": 96.7,
        "startTimestamp": null,
        "endTimestamp": null
      }
    ]
  }
}
```

### Notes for AI usage

- Use `interval` for chart downsampling.
- Use `startDate` and `endDate` for bounded history queries.
- `data.waterLevel` can be an empty array when no rows match the query window.
- For one metric only, prefer `/history/{variable}` to reduce payload size.

---

## 5. Get Water Level History for a Station, Single Variable

```http
GET /water-level/station/{stationId}/history/{variable}
```

Returns compact time-series data for a single water-level metric.

Use this when rendering one line chart or analyzing one field such as `calculatedWaterLevel`, `rawMode`, `median`, or `maximum`.

### Path parameters

| Name | Type | Required | Description |
|---|---|---:|---|
| `stationId` | string | Yes | Station hashid. |
| `variable` | string | Yes | Case-sensitive camelCase metric name. |

### Allowed variables

```txt
sampleInterval
sampleCount
filteredSampleCount
spikeCount
minimum
maximum
rawMode
calculatedWaterLevel
median
frequentRangeLow
frequentRangeHigh
estimatedMovAvg
distance
```

`distance` is an alias for `calculatedWaterLevel` in queries.

### Query parameters

| Name | Type | Required | Description |
|---|---|---:|---|
| `skip` | number | No | Offset. Defaults to `0`. |
| `take` | number | No | Defaults to `10` when omitted. Use a larger value with date windows for charts. |
| `interval` | number | No | Aggregation bucket size in minutes. Allowed: `1`, `15`, `30`, `60`, `180`, `360`, `720`, `1440`. |
| `startDate` | string | No | ISO start of window. Defaults to epoch if omitted. |
| `endDate` | string | No | ISO end of window. Defaults to now if omitted. |

### Response data shape

```ts
type WaterLevelVariableHistoryResponse = {
  station: StationInfo;
  waterLevel: VariableReading[];
};
```

Important: the response key is `waterLevel`, not `telemetry`.

### Example request

```ts
const stationId = 'st_abc123';
const variable = 'calculatedWaterLevel';

const params = new URLSearchParams({
  skip: '0',
  take: '500',
  startDate: '2026-05-01',
  endDate: '2026-05-04',
});

const response = await fetch(
  `https://api.kloudtechsea.com/api/v1/water-level/station/${stationId}/history/${variable}?${params}`,
  {
    headers: {
      'x-kloudtrack-key': 'YOUR_API_KEY_HERE',
    },
  }
);

const json = await response.json();

const chartPoints = [...json.data.waterLevel]
  .reverse()
  .map((point) => ({
    x: point.recordedAt,
    y: point.value,
  }));
```

### Example response

```json
{
  "success": true,
  "message": "Water level history retrieved successfully",
  "data": {
    "station": {
      "id": "st_abc123",
      "stationName": "River gauge 01",
      "stationType": "RIVERLEVEL"
    },
    "waterLevel": [
      {
        "id": 5002,
        "recordedAt": "2026-05-04T09:00:00.000Z",
        "createdAt": "2026-05-04T09:00:05.000Z",
        "value": 101.5
      }
    ]
  }
}
```

### Common errors

| Status | Meaning |
|---:|---|
| 400 | Invalid variable name or invalid interval. |

### Notes for AI usage

- `variable` is case-sensitive.
- Use `calculatedWaterLevel` for the main derived river/water-level value.
- Use `rawMode` if the AI needs the raw sensor aggregate mode.
- Use `minimum`, `maximum`, and `median` for distribution/quality views.
- Use `sampleCount`, `filteredSampleCount`, and `spikeCount` for data quality diagnostics.

---

## Recommended Usage Patterns

### Current overview dashboard

Use:

```http
GET /water-level/dashboard
```

Reason:

- One request returns all readable stations.
- Includes latest reading per station.
- Handles stations with no readings using `waterLevel: null`.

### Single station card/detail page

Use:

```http
GET /water-level/station/{stationId}/current
```

Reason:

- Simple current-state lookup.
- Good for detail pages and alert checks.

### Historical chart with full context

Use:

```http
GET /water-level/station/{stationId}/history?interval=60&startDate=...&endDate=...
```

Reason:

- Gives full aggregate rows.
- Useful when the chart/tool needs multiple fields like `calculatedWaterLevel`, `rawMode`, `minimum`, and `maximum`.

### Historical chart with one metric only

Use:

```http
GET /water-level/station/{stationId}/history/calculatedWaterLevel?take=500&startDate=...&endDate=...
```

Reason:

- Smaller payload.
- Easier chart transformation.
- Response points are `{ id, recordedAt, createdAt, value }`.

---

## AI Implementation Rules

When an AI uses these endpoints, follow these rules:

1. Always include the `x-kloudtrack-key` header.
2. Do not expose the API key in client-side public code unless this is intentionally a trusted/internal environment.
3. Treat station IDs as string hashids, for example `st_abc123`.
4. Treat water-level record IDs as numeric IDs, for example `5001`.
5. Do not confuse `/water-level/record/{id}` with `/water-level/station/{stationId}/current`.
6. Always handle `waterLevel: null` from the dashboard endpoint.
7. Always handle `404` from the current endpoint because a station may have no water-level rows yet.
8. Use date windows for history queries to avoid large payloads.
9. Reverse history arrays client-side when plotting chronological charts.
10. Prefer the single-variable history endpoint when only one field is needed.
11. Use `calculatedWaterLevel` as the primary display metric unless the user explicitly asks for raw sensor values.
12. Remember that `distance` is an alias for `calculatedWaterLevel` in single-variable history queries.

---

## Minimal Fetch Helper

```ts
const BASE_URL = 'https://api.kloudtechsea.com/api/v1';

async function kloudtrackGet<T>(path: string, apiKey: string): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, {
    method: 'GET',
    headers: {
      'x-kloudtrack-key': apiKey,
    },
  });

  const json = await response.json();

  if (!response.ok) {
    throw new Error(
      json?.error?.details ||
      json?.message ||
      `KloudTrack API error: ${response.status}`
    );
  }

  return json.data as T;
}
```

### Example: load dashboard

```ts
const dashboard = await kloudtrackGet<WaterLevelDashboardResponse>(
  '/water-level/dashboard',
  'YOUR_API_KEY_HERE'
);
```

### Example: load current water level

```ts
const current = await kloudtrackGet<CurrentWaterLevelResponse>(
  '/water-level/station/st_abc123/current',
  'YOUR_API_KEY_HERE'
);
```

### Example: load calculated water-level history

```ts
const params = new URLSearchParams({
  take: '500',
  startDate: '2026-05-01',
  endDate: '2026-05-04',
});

const history = await kloudtrackGet<WaterLevelVariableHistoryResponse>(
  `/water-level/station/st_abc123/history/calculatedWaterLevel?${params}`,
  'YOUR_API_KEY_HERE'
);
```
