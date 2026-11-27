<!--
Copyright Advanced Micro Devices, Inc.

SPDX-License-Identifier: MIT
-->

# Security Policy

## Reporting a vulnerability

If you have discovered a security vulnerability in amd/skillscope, please do
NOT report it publicly using a GitHub issue. Instead, report it via the AMD
Product Security website:

[https://www.amd.com/en/resources/product-security.html](https://www.amd.com/en/resources/product-security.html)

In your report, please include:

- the affected component: the `skillscope` CLI, the GitHub Action
  ([action.yml](action.yml)), or one of the reusable workflows under
  [.github/workflows/](.github/workflows/);
- the Skillscope version or the ref your workflow pins;
- steps to reproduce, with a minimal repo or dataset if possible;
- the potential impact, and any mitigation or workaround you know of.

Skillscope runs agents with real credentials, on your machine and in CI. A
credential or other secret showing up where it should not, such as a report, a
job summary, a log or an artifact, is a security problem: report it here, and
do not paste the secret into the report.
