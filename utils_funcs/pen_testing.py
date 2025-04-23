import requests
import socket
# import nmap  # Unused
# import sys   # Unused
# import paramiko # Unused
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Any
import logging
from urllib.parse import urljoin # Added for robust URL handling

# --- Configuration ---
DEFAULT_TIMEOUT = 5 # Default timeout for network requests in seconds
DEFAULT_MAX_WORKERS = 100 # Default max workers for thread pools

# --- Setup Logging ---
# Configure logging in the main block if run as script,
# otherwise rely on calling code to configure.
# logger = logging.getLogger(__name__) # Get logger for the module

class SecurityTester:
    """
    A collection of ethical penetration testing tools for authorized security assessments.
    Focuses on network-level checks.
    """
    def __init__(self, target_host: str, logger: Optional[logging.Logger] = None, timeout: int = DEFAULT_TIMEOUT):
        self.target = target_host
        # Removed <sup> tag
        self.description = "A security testing toolkit with clean and straightforward syntax, crucial for time-sensitive security assessments."
        self.logger = logger or logging.getLogger(__name__)
        self.timeout = timeout
        self.target_ip = self._resolve_target() # Resolve IP once

    def _resolve_target(self) -> Optional[str]:
        """Resolves hostname to IP address."""
        try:
            ip_address = socket.gethostbyname(self.target)
            self.logger.info(f"Resolved {self.target} to {ip_address}")
            return ip_address
        except socket.gaierror:
            self.logger.error(f"Could not resolve hostname: {self.target}")
            return None

    def port_scanner(self, start_port: int, end_port: int, max_workers: int = DEFAULT_MAX_WORKERS) -> List[int]:
        """
        Scan port range on the target host using threading.

        Args:
            start_port (int): Starting port number.
            end_port (int): Ending port number.
            max_workers (int): Maximum concurrent threads for scanning.

        Returns:
            List[int]: Sorted list of open TCP ports.
        """
        if not self.target_ip:
            self.logger.error("Cannot perform port scan, target IP not resolved.")
            return []

        open_ports: List[int] = []
        ports_to_scan = range(start_port, end_port + 1)
        self.logger.info(f"Starting port scan on {self.target_ip} for ports {start_port}-{end_port}...")

        def scan_single_port(port: int) -> None:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(self.timeout)
                    # connect_ex returns 0 on success (port open)
                    result = sock.connect_ex((self.target_ip, port))
                    if result == 0:
                        self.logger.info(f"Port {port} is open on {self.target_ip}")
                        open_ports.append(port)
            except socket.timeout:
                # Timeout usually means filtered or slow response, treat as closed/filtered
                self.logger.debug(f"Timeout scanning port {port} on {self.target_ip}")
            except OSError as e:
                # Handle other potential socket errors (e.g., network unreachable)
                self.logger.debug(f"Error scanning port {port} on {self.target_ip}: {e}")
            # No explicit close needed with 'with' statement

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(scan_single_port, ports_to_scan)

        self.logger.info(f"Port scan completed for {self.target_ip}. Found {len(open_ports)} open ports.")
        return sorted(open_ports)

    def ssl_security_check(self) -> Dict[str, Any]:
        """
        Check basic SSL/TLS configuration by attempting a secure connection.
        Verifies if the certificate is trusted by the system's default store.

        Returns:
            Dict[str, Any]: Dictionary indicating if secure and any error message.
        """
        url = f"https://{self.target}"
        self.logger.info(f"Checking SSL/TLS configuration for {self.target} via {url}")
        try:
            # verify=True is default, ensures cert validation
            response = requests.get(url, verify=True, timeout=self.timeout)
            response.raise_for_status() # Check for HTTP errors
            # If request succeeds without SSLError, consider it basically secure from this check's perspective
            self.logger.info(f"SSL/TLS check for {self.target} successful (Status: {response.status_code}).")
            return {
                "secure": True,
                # Removed unreliable 'Server-Cert' header check
                "cert_info": "Basic check passed (Trusted Cert/Successful Connection)"
            }
        except requests.exceptions.SSLError as e:
            self.logger.warning(f"SSL/TLS Error for {self.target}: {e}")
            return {
                "secure": False,
                "error": f"SSLError: {e}"
            }
        except requests.exceptions.Timeout:
            self.logger.error(f"Timeout during SSL check for {self.target}")
            return {
                "secure": False,
                "error": "Timeout during connection attempt"
            }
        except requests.exceptions.RequestException as e:
            # Catch other request errors (ConnectionError, etc.)
            self.logger.error(f"Request failed during SSL check for {self.target}: {e}")
            return {
                "secure": False,
                "error": f"RequestException: {e}"
            }
            # Removed <sup> tag from original error return comment

    def directory_scanner(self, wordlist: List[str], base_url_scheme: str = "https", max_workers: int = 50) -> List[str]:
        """
        Scan for existing directories/files using a wordlist via threading.

        Args:
            wordlist (List[str]): List of paths/filenames to check.
            base_url_scheme (str): Scheme to use ('http' or 'https'). Defaults to 'https'.
            max_workers (int): Maximum concurrent threads for scanning.

        Returns:
            List[str]: Sorted list of URLs found with a 200 OK status.
        """
        found_dirs: List[str] = []
        base_url = f"{base_url_scheme}://{self.target}/" # Ensure trailing slash
        self.logger.info(f"Starting directory scan on {base_url} with {len(wordlist)} paths...")

        def check_path(path: str) -> None:
            path = path.strip()
            if not path:
                return # Skip empty lines

            # Use urljoin for safer path combination
            url = urljoin(base_url, path)
            try:
                # Use HEAD request first to be lighter? Or GET to check content? Sticking with GET for now.
                # allow_redirects=False is often useful for directory scanning
                response = requests.get(url, timeout=self.timeout, allow_redirects=False, headers={'User-Agent': 'Security-Test-Scanner/1.0'})
                # Check for 200 OK. Could also check for 403 (Forbidden) as interesting.
                if response.status_code == 200:
                    self.logger.info(f"Found: {url} (Status: {response.status_code})")
                    found_dirs.append(url)
                # Optionally log other interesting status codes
                # elif response.status_code == 403:
                #     self.logger.info(f"Potential (Forbidden): {url} (Status: {response.status_code})")
                else:
                     self.logger.debug(f"Checked: {url} (Status: {response.status_code})")

            except requests.exceptions.Timeout:
                self.logger.debug(f"Timeout checking path: {url}")
            except requests.exceptions.RequestException as e:
                self.logger.debug(f"Error checking path {url}: {e}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
             list(executor.map(check_path, wordlist)) # Use list() to ensure all futures complete

        self.logger.info(f"Directory scan completed for {base_url}. Found {len(found_dirs)} accessible paths (Status 200).")
        return sorted(found_dirs)
        # Removed <sup> tag from original comment

class WebSecurityTester:
    """
    Web application security testing tools (SQLi, XSS).
    Requires appropriate authorization. Use with caution.
    """
    def __init__(self, target_url: str, logger: Optional[logging.Logger] = None, timeout: int = DEFAULT_TIMEOUT):
        # Ensure target_url includes scheme (http/https)
        if not target_url.startswith(('http://', 'https://')):
             raise ValueError("target_url must include scheme (http:// or https://)")
        self.target_url = target_url
        # Removed <sup> tag
        self.description = "Common tasks for web security assessment including basic SQLi and XSS checks."
        self.user_agent = "Security-Test-Agent/1.0" # Updated User-Agent slightly
        self.logger = logger or logging.getLogger(__name__)
        self.timeout = timeout
        self.headers = {'User-Agent': self.user_agent}

    def sql_injection_test(self, endpoint: str, method: str = 'POST', params_or_data: Dict[str, str] = None, injection_param: str = None, payload: str = "' OR '1'='1") -> Dict[str, Any]:
        """
        Test for basic SQL injection vulnerabilities.
        NOTE: This is a very basic check and may produce false positives/negatives.

        Args:
            endpoint (str): The specific path/endpoint relative to target_url (e.g., '/login').
            method (str): HTTP method ('GET' or 'POST'). Defaults to 'POST'.
            params_or_data (Dict[str, str]): Dictionary of other parameters/data needed for the request.
            injection_param (str): The specific parameter name to inject the payload into.
            payload (str): SQL injection payload. Defaults to a common basic payload.

        Returns:
            Dict[str, Any]: Dictionary with status, response size, potential vulnerability flag, and error.
        """
        if not injection_param or not params_or_data:
             self.logger.error("SQL injection test requires params_or_data and injection_param.")
             return {"error": "Missing required arguments: params_or_data and injection_param"}

        full_url = urljoin(self.target_url, endpoint)
        test_data = params_or_data.copy()
        test_data[injection_param] = payload # Inject payload

        self.logger.info(f"Attempting basic SQLi test on {full_url} with param '{injection_param}'")

        try:
            if method.upper() == 'POST':
                response = requests.post(full_url, data=test_data, headers=self.headers, timeout=self.timeout)
            elif method.upper() == 'GET':
                response = requests.get(full_url, params=test_data, headers=self.headers, timeout=self.timeout)
            else:
                return {"error": f"Unsupported HTTP method: {method}"}

            response.raise_for_status() # Check for HTTP errors

            # Basic check: Look for common SQL error patterns or unexpected keywords. Highly unreliable.
            potential_vulnerability = any(err in response.text.lower() for err in ["sql syntax", "mysql", "syntax error", "unclosed quotation", "odbc", "ora-"])
            # Alternative basic check (original): "table" in response.text.lower()

            return {
                "status_code": response.status_code,
                "response_size": len(response.text),
                "potential_vulnerability": potential_vulnerability,
                "error": None
            }
        except requests.exceptions.RequestException as e:
            self.logger.error(f"SQL injection test failed for {injection_param}: {e}")
            return {"error": str(e), "potential_vulnerability": False} # Assume not vulnerable on error

    def xss_scanner(self, endpoint: str, parameters: List[str]) -> Dict[str, bool]:
        """
        Test for basic reflected XSS vulnerabilities in specified GET parameters.
        NOTE: This is a very basic check for reflected XSS only.

        Args:
            endpoint (str): The specific path/endpoint relative to target_url (e.g., '/search').
            parameters (List[str]): List of GET parameter names to test.

        Returns:
            Dict[str, bool]: Dictionary mapping parameter name to potential vulnerability (True/False).
        """
        results = {}
        test_payloads = [
            "<script>alert('XSS_Test')</script>",
            "<img src=x onerror=alert('XSS_Test')>",
            "javascript:alert('XSS_Test')",
            "'\"--><script>alert('XSS_Test')</script>" # Added another variation
        ]
        full_url = urljoin(self.target_url, endpoint)
        self.logger.info(f"Starting basic XSS scan on {full_url} for params: {parameters}")

        for param in parameters:
            results[param] = False # Default to not vulnerable
            for payload in test_payloads:
                try:
                    response = requests.get(
                        full_url,
                        params={param: payload},
                        headers=self.headers,
                        timeout=self.timeout
                    )
                    # Basic check: Does the exact payload appear in the response? Very limited.
                    if payload in response.text:
                        self.logger.warning(f"Potential XSS reflection found for param '{param}' with payload: {payload}")
                        results[param] = True
                        break # Found potential XSS for this param, move to next param
                except requests.exceptions.RequestException as e:
                    self.logger.error(f"XSS test failed for param '{param}' with payload '{payload}': {e}")
                    # Keep results[param] as False on error
        return results

def security_report_generator(test_results: Dict[str, Any]) -> str:
    """
    Generate a formatted security report from test results.
    NOTE: This function expects results to be processed into a consistent
          format before being passed in (e.g., {'Test Name': {'vulnerable': bool, 'details': str}}).
          The raw output from tester methods needs adaptation.
    """
    report = "Security Assessment Report\n"
    report += "=" * 30 + "\n\n"

    for test_name, result_data in test_results.items():
        report += f"Test: {test_name}\n"
        if isinstance(result_data, dict):
            status = 'Vulnerable' if result_data.get('vulnerable', False) else 'Secure/Not Found'
            details = result_data.get('details', 'N/A')
            report += f"Status: {status}\n"
            report += f"Details: {details}\n"
        else:
            # Handle simpler results if needed, or mark as unknown format
             report += f"Result: {result_data} (Format requires processing)\n"
        report += "-" * 30 + "\n"

    return report

# Example usage with safety checks
if __name__ == "__main__":
    # --- Basic Logging Setup ---
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')
    main_logger = logging.getLogger(__name__)

    print("*" * 60)
    print("          PENETRATION TESTING SCRIPT")
    print("*" * 60)
    print("\nWARNING: Unauthorized testing is illegal and unethical.")
    print("Ensure you have explicit permission before scanning any target.\n")

    target = input("Enter authorized target host (e.g., example.com or IP): ")
    if not target:
        print("No target specified. Exiting.")
        exit(1)

    # --- Instantiate Testers ---
    # Pass the logger instance
    tester = SecurityTester(target, logger=main_logger)
    # Assume HTTPS for web tests, adjust if needed
    web_target_url = f"https://{target}"
    try:
        web_tester = WebSecurityTester(web_target_url, logger=main_logger)
    except ValueError as e:
        main_logger.error(f"Failed to initialize WebSecurityTester: {e}")
        web_tester = None # Cannot perform web tests

    # --- Run Basic Security Checks ---
    results_summary = {} # To potentially use with report generator later

    main_logger.info(f"--- Starting Network Scans for {target} ---")
    if tester.target_ip: # Proceed only if target resolved
        # 1. Port Scan (Example: Common web ports + some others)
        # common_ports = [21, 22, 23, 25, 53, 80, 110, 135, 139, 143, 443, 445, 993, 995, 1723, 3306, 3389, 5900, 8080]
        common_ports = list(range(1, 1025)) # Scan well-known ports
        print(f"\n[+] Performing Port Scan (Ports {min(common_ports)}-{max(common_ports)})...")
        open_ports = tester.port_scanner(min(common_ports), max(common_ports))
        print(f"[+] Open Ports Found: {open_ports}")
        results_summary['Port Scan'] = {'vulnerable': bool(open_ports), 'details': f'Open ports: {open_ports}'}

        # 2. SSL Check
        print(f"\n[+] Performing SSL/TLS Check for {target}...")
        ssl_check = tester.ssl_security_check()
        print(f"[+] SSL Security Status: {'Secure' if ssl_check.get('secure') else 'Insecure/Error'}")
        if ssl_check.get('error'):
            print(f"    Reason: {ssl_check['error']}")
        results_summary['SSL Check'] = {'vulnerable': not ssl_check.get('secure'), 'details': ssl_check.get('error') or ssl_check.get('cert_info')}

        # 3. Directory Scan (Example - requires a wordlist file)
        wordlist_file = "common_dirs.txt" # Example wordlist filename
        try:
            print(f"\n[+] Attempting Directory Scan (using '{wordlist_file}')...")
            with open(wordlist_file, 'r') as f:
                wordlist = f.read().splitlines()

            if wordlist:
                # Determine scheme based on open ports if possible
                scheme = "https" if 443 in open_ports else "http"
                if not (80 in open_ports or 443 in open_ports):
                     print(f"[*] Ports 80/443 not detected open, defaulting directory scan to HTTPS for {target}")
                     scheme = "https" # Default guess

                found_paths = tester.directory_scanner(wordlist, base_url_scheme=scheme)
                if found_paths:
                    print(f"[+] Found Accessible Paths (Status 200):")
                    for path in found_paths:
                        print(f"  - {path}")
                else:
                    print("[-] No accessible paths found from the wordlist with status 200.")
                results_summary['Directory Scan'] = {'vulnerable': bool(found_paths), 'details': f'Found paths: {found_paths}'}
            else:
                print(f"[*] Wordlist file '{wordlist_file}' is empty. Skipping directory scan.")

        except FileNotFoundError:
            print(f"[-] Wordlist file '{wordlist_file}' not found. Skipping directory scan.")
            print(f"    (Create a file named '{wordlist_file}' with one path per line, e.g., 'admin', 'login.php')")
        except Exception as e:
             main_logger.error(f"An error occurred during directory scan setup: {e}")

    else:
        main_logger.error(f"Target {target} could not be resolved. Skipping network scans.")


    # --- Run Basic Web App Checks (if WebTester initialized) ---
    if web_tester:
        main_logger.info(f"--- Starting Web Application Scans for {web_target_url} ---")

        # 4. SQL Injection Test (Example - requires knowing endpoint and params)
        # print("\n[+] Performing Basic SQL Injection Test (Example)...")
        # print("[!] This requires knowing a valid endpoint and parameter name.")
        # # Replace with actual endpoint and parameters for the authorized target
        # sqli_endpoint = '/login'
        # sqli_params = {'username': 'testuser', 'password': 'fakepassword'} # Example other params
        # sqli_inject_param = 'username' # Parameter to inject into
        # sqli_result = web_tester.sql_injection_test(sqli_endpoint, method='POST', params_or_data=sqli_params, injection_param=sqli_inject_param)
        # print(f"[+] SQLi Test Result for param '{sqli_inject_param}':")
        # if sqli_result.get("error"):
        #     print(f"    Error: {sqli_result['error']}")
        # else:
        #     print(f"    Potential Vulnerability Detected: {sqli_result.get('potential_vulnerability', False)}")
        #     print(f"    Status Code: {sqli_result.get('status_code')}, Response Size: {sqli_result.get('response_size')}")
        # results_summary['SQL Injection'] = {'vulnerable': sqli_result.get('potential_vulnerability', False), 'details': f'Param: {sqli_inject_param}, Error: {sqli_result.get("error")}'}


        # 5. XSS Scan (Example - requires knowing endpoint and params)
        # print("\n[+] Performing Basic Reflected XSS Scan (Example)...")
        # print("[!] This requires knowing a valid endpoint and GET parameter names.")
        # # Replace with actual endpoint and parameters for the authorized target
        # xss_endpoint = '/search'
        # xss_params_to_test = ['query', 'category'] # Example parameters
        # xss_results = web_tester.xss_scanner(xss_endpoint, xss_params_to_test)
        # print(f"[+] XSS Scan Results:")
        # vulnerable_params = [p for p, v in xss_results.items() if v]
        # if vulnerable_params:
        #     print(f"    Potential XSS found in parameters: {vulnerable_params}")
        # else:
        #     print("    No basic reflected XSS detected in tested parameters.")
        # results_summary['XSS Scan'] = {'vulnerable': bool(vulnerable_params), 'details': f'Tested params: {xss_params_to_test}, Vulnerable: {vulnerable_params}'}

    else:
         main_logger.warning("WebSecurityTester not initialized. Skipping web application scans.")


    # --- Generate Report (Example) ---
    # print("\n--- Generating Basic Report ---")
    # report = security_report_generator(results_summary)
    # print(report)
    # # Optionally write report to file
    # # with open(f"security_report_{target}.txt", "w") as f:
    # #     f.write(report)

    print("\n--- Security Assessment Finished ---")

