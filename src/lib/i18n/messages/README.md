# Translation Messages

Use TypeScript message files instead of JSON when adding app-maintained locales.

To add a new locale, create a full message file next to `en.ts`:

```ts
import type { Messages } from "@/lib/i18n/translations";

export const fil = {
  common: {
    // Every key from en.ts must exist here.
  },
} satisfies Messages;
```

Do not use `{ ...en }` for real translated locales. Spreading English hides missing translations instead of forcing complete equivalents.
