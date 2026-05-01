#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM server connection.

Part of CASCADE-VLM (CVPR 2026 ACCIDENT Challenge submission).
"""

from openai import OpenAI


def get_client(url):
    from openai import OpenAI
    return OpenAI(base_url=url, api_key="dummy")


def check_server(url):
    try:
        c = get_client(url)
        c.models.list()
        return True
    except Exception as e:
        logger.warning(f"서버 체크 실패: {e}")
        return False


# ─────────────────────────────────────────────
