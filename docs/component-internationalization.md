# Component Internationalization

Use this guide when adding or updating a component that contains user-visible
text. This application uses `next-intl` and supports English (`en`) and Filipino
(`fil`).

Station-only changes in `src/lib/constants/stations.json` do not require
translations. Station names and addresses come from the Kloudtrack API.

## Text that must be translated

Do not hard-code user-facing text in a component. Translation keys are required
for:

- Headings, descriptions, labels, buttons, and links
- Form placeholders and validation messages
- Loading, empty, warning, success, and error states
- Tooltips and dialog content
- Accessibility text such as `aria-label`, `alt`, and screen-reader-only copy

Developer-only logs, code comments, identifiers, and API values do not require
translation keys.

## Add the message keys

Add the same nested key structure to both message files:

```text
src/lib/i18n/messages/en.ts
src/lib/i18n/messages/fil.ts
```

Use `en.ts` as the message structure. For example:

```ts
dashboard: {
  stationStatus: {
    title: "Station Status",
    unavailable: "Station data is unavailable.",
  },
},
```

Add the matching Filipino keys in `fil.ts`:

```ts
dashboard: {
  stationStatus: {
    title: "Kalagayan ng Station",
    unavailable: "Hindi available ang datos ng station.",
  },
},
```

The key paths must be identical:

```text
English:  dashboard.stationStatus.title
Filipino: dashboard.stationStatus.title
```

`fil.ts` uses `satisfies Messages`, where `Messages` is derived from `en.ts`.
This allows TypeScript to report missing keys or a structure that does not
match English.

Do not spread or copy the English dictionary into `fil.ts`. Every supported
locale must provide its own translated value.

## Reference keys in a client component

Client components should import `useTranslations` from `next-intl`:

```tsx
"use client";

import { useTranslations } from "next-intl";

export function StationStatus() {
  const t = useTranslations("dashboard.stationStatus");

  return (
    <section>
      <h2>{t("title")}</h2>
      <p>{t("unavailable")}</p>
    </section>
  );
}
```

The namespace and local key combine into the complete message path:

```text
useTranslations("dashboard.stationStatus") + t("title")
                                      ↓
dashboard.stationStatus.title
```

An existing namespace can also be referenced from the root:

```tsx
const t = useTranslations();

return <h2>{t("dashboard.stationStatus.title")}</h2>;
```

Prefer a scoped namespace when a component uses several messages from the same
section.

## Reference keys in an async server component

Async server components should import `getTranslations` from
`next-intl/server`:

```tsx
import { getTranslations } from "next-intl/server";

export default async function StationStatusPage() {
  const t = await getTranslations("dashboard.stationStatus");

  return <h1>{t("title")}</h1>;
}
```

## Variables and placeholders

Use message placeholders for dynamic values:

```ts
// en.ts
lastUpdated: "Last updated at {time}",

// fil.ts
lastUpdated: "Huling na-update noong {time}",
```

Pass the value from the component:

```tsx
<p>{t("lastUpdated", {time: formattedTime})}</p>
```

Placeholder names must be identical in every locale. If English uses `{time}`,
Filipino must also use `{time}`.

## Update an existing component

When changing existing copy:

1. Find the key currently referenced by the component.
2. Update that key in both `en.ts` and `fil.ts`.
3. If the key is renamed, update every component reference to the new path.
4. Search for the old key to ensure no stale references remain:

   ```powershell
   rg "oldTranslationKey" src
   ```

5. Do not rename a shared key only for one component. Create a component-specific
   key when the wording has a different meaning in that component.

## Validate the change

1. Run the TypeScript check:

   ```powershell
   npm run type-check
   ```

2. Open the affected component and switch between English and Filipino using
   the language selector.
3. Confirm every new or changed message switches language.
4. Confirm no raw key such as `dashboard.stationStatus.title` appears in the UI.
5. Check responsive layouts because translated text may be longer.
6. Check accessibility text, tooltips, loading states, empty states, and error
   states—not only the component's primary content.

For instructions on adding an entirely new locale, see
`src/lib/i18n/messages/README.md`.
