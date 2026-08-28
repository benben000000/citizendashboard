# Station Fetch Configuration

To add or remove stations from the public dashboard, update only:

```text
src/lib/constants/stations.json
```

Do not update the services, pages, station selector, or map.

## Choose the station list

The JSON file has two separate lists:

- `weather.stationIdToFetch` for weather stations
- `waterLevel.stationIdToFetch` for water-level stations

## Add a weather station

Add this object inside `weather.stationIdToFetch`:

```json
{
  "stationId": "STATION_PUBLIC_ID",
  "contactNumber": "",
  "email": "",
  "location": "location-name"
}
```

Replace:

- `STATION_PUBLIC_ID` with the station's exact public ID. IDs are
  case-sensitive.
- `location-name` with a unique lowercase location slug, such as `san-jose`.
- `contactNumber` and `email` with the station's contact details, or leave them
  as empty strings when unavailable.

## Add a water-level station

Add this object inside `waterLevel.stationIdToFetch`:

```json
{
  "stationId": "STATION_PUBLIC_ID",
  "contactNumber": "",
  "email": "",
  "location": "location-name",
  "referenceThreshold": 780
}
```

Replace the values as described above. Set `referenceThreshold` to the approved
bridge/reference height in centimeters. Remove that field when no approved
threshold is available.

## Remove a station

Delete the station's complete object from the correct `stationIdToFetch` list.
If the station exists in both lists, remove it from both only when it should
disappear from both dashboards.

Remember to fix the comma between the remaining JSON objects.

## Rules to remember

- The order of the objects controls the dashboard station order.
- The first available object becomes the default station.
- Do not add the station name, address, coordinates, or readings. Those values
  come from the Kloudtrack API.
- Keep at least one station in each list. An empty `stationIdToFetch` list
  currently means show all available stations, not show none.
- After updating the file, rebuild and redeploy the application.

## Check the JSON

Run this command after editing:

```powershell
Get-Content -Raw src/lib/constants/stations.json |
  ConvertFrom-Json |
  Out-Null
```

No output means the JSON is valid.
