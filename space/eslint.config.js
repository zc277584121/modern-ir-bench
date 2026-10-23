import js from "@eslint/js";
import globals from "globals";

export default [
  { ignores: ["data/**"] },
  js.configs.recommended,
  { files: ["*.js"], languageOptions: { globals: globals.browser } },
  { files: ["*.test.mjs"], languageOptions: { globals: globals.node } },
];
