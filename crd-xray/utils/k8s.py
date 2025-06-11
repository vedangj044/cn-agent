import logging
from kubernetes.client import CustomObjectsApi, CoreV1Api
from typing import Dict, Any
from datetime import datetime, timezone
from models import CRD
import json


def get_all_crds(custom_api: CustomObjectsApi):
    """Fetch all CRDs from the cluster"""
    logging.info("Fetching all CRDs from the cluster...")
    try:
        crds = custom_api.list_cluster_custom_object(
            group="apiextensions.k8s.io",
            version="v1",
            plural="customresourcedefinitions",
        )
        logging.info(f"Found {len(crds['items'])} CRDs")
        return crds["items"]
    except Exception as e:
        logging.error(f"Failed to fetch CRDs: {str(e)}")
        raise


def get_crd_events(crd_kind: str, last_fetched: datetime | None, core_api: CoreV1Api):
    """Fetch events related to a specific CRD"""
    logging.info(f"Fetching events for CRD: {crd_kind}")
    try:
        events = core_api.list_event_for_all_namespaces(
            field_selector=f"involvedObject.kind={crd_kind}"
        )

        if last_fetched is None:
            filtered_events = events.items
        else:
            filtered_events = [
                event
                for event in events.items
                if event.last_timestamp
                and event.last_timestamp.replace(tzinfo=timezone.utc) > last_fetched
            ]

        logging.info(f"Found {len(filtered_events)} events for CRD {crd_kind}")
        return filtered_events
    except Exception as e:
        logging.error(f"Failed to fetch events for CRD {crd_kind}: {str(e)}")
        raise


def get_crd_resources(crd: Dict[str, Any], custom_api: CustomObjectsApi):
    """Fetch all resources for a specific CRD"""
    crd_name = crd["metadata"]["name"]
    logging.info(f"Fetching resources for CRD: {crd_name}")
    try:
        group = crd["spec"]["group"]
        version = crd["spec"]["versions"][0]["name"]
        plural = crd["spec"]["names"]["plural"]

        resources = custom_api.list_cluster_custom_object(
            group=group, version=version, plural=plural
        )
        logging.info(f"Found {len(resources['items'])} resources for CRD {crd_name}")
        return resources["items"]
    except Exception as e:
        logging.error(f"Failed to fetch resources for CRD {crd_name}: {str(e)}")
        raise


def get_controller(crd_obj: CRD, core_api: CoreV1Api):
    pods = core_api.list_pod_for_all_namespaces().items
    controller_pod_obj = next(
        (p for p in pods if p.metadata.name == crd_obj.controller_name), None
    )

    if controller_pod_obj:
        return controller_pod_obj
    logging.warning(f"Controller pod {crd_obj.controller_name} no longer exists")
    return None


def get_controller_logs(crd_obj: CRD, namespace: str, core_api: CoreV1Api) -> str | None:
    try:
        current_logs = core_api.read_namespaced_pod_log(
            name=crd_obj.controller_name,
            namespace=namespace,
            tail_lines=100,
        )
    except Exception as e:
        logging.error(
            f"Failed to fetch logs for pod {crd_obj.controller_name}: {str(e)}"
        )
        return "None"
    return current_logs
