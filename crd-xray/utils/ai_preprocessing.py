import logging
from langchain_anthropic import ChatAnthropic
from typing import List, Any, Dict
import json
from dataclasses import dataclass


@dataclass
class PodInfo:
    name: str
    namespace: str
    labels: Dict[str, str]
    annotations: Dict[str, str]
    container_images: List[str]
    status: str
    creation_timestamp: str


def extract_pod_info(pod) -> PodInfo:
    """Extract relevant information from a Kubernetes pod object"""
    return PodInfo(
        name=pod.metadata.name,
        namespace=pod.metadata.namespace,
        labels=pod.metadata.labels or {},
        annotations=pod.metadata.annotations or {},
        container_images=[container.image for container in pod.spec.containers],
        status=pod.status.phase,
        creation_timestamp=(
            pod.metadata.creation_timestamp.isoformat()
            if pod.metadata.creation_timestamp
            else ""
        ),
    )


def identify_controller(
    crd_name: str, crd_manifest: str, pods: List[Any], model_name: str, api_key: str
) -> str:
    """Identify the controller pod for a CRD using LLM"""
    llm = ChatAnthropic(
        model_name=model_name, api_key=api_key, timeout=60, stop=None, temperature=0
    )

    pod_infos = [extract_pod_info(pod) for pod in pods]

    prompt = f"""
    Given the following CRD manifest and list of pods, identify which pod is most likely to be the controller for this CRD.
    Return the response in the following JSON format (do not include markdown formatting or code blocks):
    {{
        "controller_pod": "pod_name",
        "confidence": 0.95,
        "reasoning": "explanation of why this pod is likely the controller"
    }}
    
    CRD Name: {crd_name}
    CRD Manifest: {crd_manifest}
    
    Available Pods:
    {json.dumps([pod.__dict__ for pod in pod_infos], indent=2)}
    """

    logging.info(f"Sending prompt to Claude for CRD {crd_name}")
    response = llm.invoke(prompt)
    logging.info(f"Raw Claude response for {crd_name}: {response.content}")

    try:
        content = str(response.content)
        if content.startswith("```json"):
            content = content[7:]  # Remove ```json
        if content.startswith("```"):
            content = content[3:]  # Remove ```
        if content.endswith("```"):
            content = content[:-3]  # Remove ```
        content = content.strip()

        result = json.loads(content)
        logging.info(
            f"Parsed controller identification result for {crd_name}: {json.dumps(result, indent=2)}"
        )
        return result["controller_pod"]
    except Exception as e:
        logging.error(
            f"Failed to parse controller identification response for {crd_name}: {str(e)}"
        )
        logging.error(f"Cleaned content that failed to parse: {content}")
        return ""


def summarize_logs(
    controller_name: str,
    stored_logs: str,
    current_logs: str,
    model_name: str,
    api_key: str,
):

    llm = ChatAnthropic(
        model_name=model_name,
        api_key=api_key,
        timeout=60,
        stop=None,
        temperature=0,
    )

    prompt = f"""
    Analyze the following controller logs and create a structured lifecycle summary.
    The logs are from a Kubernetes controller pod that manages a Custom Resource Definition (CRD).
    
    Previous logs / previous lifecycle events:
    {stored_logs}
    
    Current logs:
    {current_logs}
    
    Please analyze these logs and provide a response in the following format:

    lifecycle_events
    <Describe the normal lifecycle events, startup, initialization, processing cycles, etc.>

    error_events
    <Describe any errors, their types, and descriptions>

    Focus on:
    1. Identifying normal lifecycle events (startup, initialization, processing cycles)
    2. Detecting and categorizing errors
    3. Understanding the controller's behavior patterns
    4. Highlighting any anomalies or concerns
    """

    try:
        logging.info(
            f"Sending prompt to Claude for Log summarization: {controller_name}"
        )
        response = llm.invoke(prompt)
        logging.info(f"Raw Claude response for {response.content}")
        return str(response.content)

    except Exception as e:
        logging.error(
            f"Failed to get response for log summarization {controller_name}: {str(e)}"
        )
    return None
