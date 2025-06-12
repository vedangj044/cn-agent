# 🩻 CRD-Xray

> *“Ever wondered what that mysterious `FooBar.custom.io` CRD actually **does**?”*
> Enter **CRD-Xray** — your AI-powered Kubernetes sidekick that breaks open black boxes of Custom Resource Definitions and their controllers.

---

## 👀 What is CRD-Xray?

**CRD-Xray** is a Kubernetes operator infused with the power of LLMs (Large Language Models). It lives inside your cluster and answers questions like:

* *“What does this CRD actually do?”*
* *“What does its controller manage?”*
* *“Why are these CRs failing mysteriously?”*

Whether it’s an internal CRD cooked up in your org or an obscure open-source one — CRD-Xray is here to help.

---

## 🚀 Features

* 🧠 **Understand Your CRDs**
  Get rich, AI-powered insights into what a CRD does, what its fields mean, and how it's used in your cluster.

* 🔍 **Lifecycle Event Summarization**
  Tired of reading 300 log lines? We summarize your controller logs into human-readable lifecycle events and error states.

* 🧩 **Agent Plugin Architecture** *(WIP)*
  Seamlessly integrates with other AI agents for an unbreakable context pipeline across your infra.

* 📊 **CRD Metrics Export** *(WIP)*
  Automatically collects and exports CRD-level metrics like number of resources, controller errors, etc.

---

## 💡 Core Idea

If you're running a modern K8s-based business, you're probably using custom CRDs and operators. These are **powerful**, but:

* New engineers can't make sense of them without trawling through old code or outdated internal docs.
* Debugging CR issues becomes a game of guess-the-error-message.

CRD-Xray fixes that. It persistently stores and indexes controller logs, CRD schemas, and real CR instances into a vector database. When queried (via API or UI), it offers **insightful**, **contextual**, and **actionable** answers.

---

## ⚠️ Disclaimer

This is an early **Proof of Concept**.
I am still laying the plumbing, sketching out the UI, and taming the agents. But I am building in the open — and feedback is welcome!

Expect big things soon 🚧

---

## 📜 License

MIT – Free to use, fork, remix, and share. Just don’t forget to ✨ star ✨ the repo if you liked it.

---

## 👋 Contribute / Say Hi

Got ideas?
Want to use it at your company?
Spotted a bug with an unpronounceable CRD?

Open an issue, drop a PR, or just reach out! Let's make Kubernetes more explainable, one CRD at a time.
