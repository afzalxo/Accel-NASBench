# Maintenance Plan

Maintainers: [Anonymous Author 1](), [Anonymous Author 2]()

This maintenance plans serves to outline plans for maintaining the project long-term, our commitment, and policies around aspects such as responding to issues and contributions.

### TODO List

The following list details our goals as part of showcasing the abilities of training proxies in efficient construction of NAS benchmarks and utilization of hardware performance metrics in search.

- [ ] Explore the applicability of approach to single-shot NAS search spaces. Search for proxy training configurations for the NASBench-201 search space, combined with single-shot search techniques such as DARTS, PC-DARTS, and GAEA. Given the popularity of single-shot search methods, showcasing the ability of training proxies to replicate the results of NASBench-201 at a significantly reduced benchmark construction cost, while achieving a high-degree of rank correlation between the results of baseline NASBench-201 and our proxified results would significantly improve the appeal of the benchmark construction approach.

- [ ] Explore the applicability of training proxy configurations to other important vision tasks, such as object detection and semantic segmentation, and more popular models such as transformers.

- [ ] Explore the impact of including low-level hardware architecture details (e.g., memory bandwidth requirements and cache sizes) into the search process. Find rules based on these hardware considerations related to which neural architectural components are preffered under which hardware constraints (e.g., high memory bandwidth prefers disabling memory-bound operations such as squeuze-excite).

- [ ] Explore efficient search techniques for training proxy configurations. Are there sophisticated search methods beyond grid search that allow a high degree of parallelism combined with a more efficient search?

- [ ] Add performance surrogates for TPUv4, KV260 FPGA, and RTX 4090 GPU.

### Long-Term Support

This project is committed to provide maintenance and support for the foreseeable future. Barring unforeseen circumstances, we plan to continue maintaining the project and addressing issues for at least the next 2 years.

Data integrity and security issues will be prioritized for fixes, even for older unsupported version branches if feasible.

### Issue Triage and Response

New issues will be triaged within 2 business days of being opened.

Issues will be tagged with priority and effort estimates.

High priority issues may get hotfixes or point releases apart from regular release cycles.

Best efforts will be made to resolve easy issues within 2 weeks.

For complex issues requiring discussions, the initial response may be to gather feedback from users and contributors. Resolution time will depend on consensus and development resources available.

### Contributions

Contributions including bug reports, feature requests, pull requests are welcomed.

New contributors will be supported by maintainers to get started and their efforts recognized.

For major new features, it is recommended to discuss the proposal early and get agreement on direction before investing heavily in development.

### Communication Channels

GitHub issues and discussions are the primary channel for support.

For time-sensitive requests, contact the maintainers via email at [Anonymous Author 1]().

Let us know if you have any other questions! We aim to provide an inclusive community and transparent process for the maintenance of this benchmark.
