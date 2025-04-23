# Network Security Testing Tools

import socket
import requests
# import nmap  # Removed unused import
import sys
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Iterable

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Try importing scapy and warn if unavailable or permission issues arise
try:
    import scapy.all as scapy
    SCAPY_AVAILABLE = True
except ImportError:
    logging.warning("Scapy library not found. Network sniffing functionality will be unavailable.")
    SCAPY_AVAILABLE = False
except OSError as e:
    logging.warning(f"Scapy initialization failed (possibly due to permissions): {e}. Network sniffing may not work.")
    SCAPY_AVAILABLE = False # Treat as unavailable if init fails

# Default User-Agent for requests
DEFAULT_USER_AGENT = "EthicalSecurityScanner/1.0"
DEFAULT_TIMEOUT = 5 # seconds

class NetworkSecurityTester:
    """
    A collection of ethical network penetration testing tools.
    Requires appropriate authorization before use against any target.
    """
    def __init__(self, user_agent: str = DEFAULT_USER_AGENT, timeout: int = DEFAULT_TIMEOUT):
        self.description = "A security testing toolkit with clean syntax for network assessments."
        self.user_agent = user_agent
        self.timeout = timeout
        self.headers = {'User-Agent': self.user_agent}

    def resolve_target(self, target: str) -> Optional[str]:
        """Resolves hostname to IP address."""
        try:
            ip_address = socket.gethostbyname(target)
            logging.info(f"Resolved {target} to {ip_address}")
            return ip_address
        except socket.gaierror:
            logging.error(f"Could not resolve hostname: {target}")
            return None

    def port_scanner(self, target: str, ports: Iterable[int]) -> List[int]:
        """
        Basic port scanner to identify open TCP ports on a target.

        Args:
            target (str): The target hostname or IP address.
            ports (Iterable[int]): An iterable of port numbers to scan (e.g., range(1, 1025)).

        Returns:
            List[int]: A sorted list of open ports found.
        """
        target_ip = self.resolve_target(target)
        if not target_ip:
            return []

        open_ports: List[int] = []
        logging.info(f"Starting port scan on {target_ip} for ports {min(ports)}-{max(ports)}...")

        def scan_port(port: int) -> None:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(self.timeout)
                    # connect_ex returns 0 on success
                    result = sock.connect_ex((target_ip, port))
                    if result == 0:
                        logging.info(f"Port {port} is open on {target_ip}")
                        open_ports.append(port)
            except socket.timeout:
                # Port is likely filtered or host is slow
                pass
            except OSError as e:
                # Handle other potential socket errors
                logging.debug(f"Error scanning port {port} on {target_ip}: {e}")
            # No return needed here, modifies list via closure

        # Use max_workers instead of max_threads
        with ThreadPoolExecutor(max_workers=100) as executor:
            executor.map(scan_port, ports)

        logging.info(f"Port scan completed for {target_ip}. Found {len(open_ports)} open ports.")
        return sorted(open_ports) # Return sorted list

    def network_sniffer(self, count: int = 10, interface: Optional[str] = None, filter_str: Optional[str] = None) -> None:
        """
        Network packet sniffer using scapy.
        NOTE: Requires scapy installation and usually root/administrator privileges.

        Args:
            count (int): Number of packets to sniff. 0 means sniff indefinitely until stopped (Ctrl+C).
            interface (Optional[str]): Specific network interface to sniff on. Default is None (all).
            filter_str (Optional[str]): BPF filter string (e.g., "tcp port 80"). Default is None (all traffic).
        """
        if not SCAPY_AVAILABLE:
            logging.error("Cannot perform network sniffing because Scapy is not available or failed to initialize.")
            return

        logging.info(f"Starting network sniffing for {count if count > 0 else 'infinite'} packets...")
        logging.warning("Network sniffing requires root/administrator privileges.")

        def packet_handler(packet) -> None:
            # Example: Print IP source and destination if available
            if packet.haslayer('IP'):
                ip_layer = packet.getlayer('IP')
                proto = ip_layer.proto # Protocol number
                protocol_name = {6: 'TCP', 17: 'UDP', 1: 'ICMP'}.get(proto, f'Other({proto})')
                log_msg = f"IP Packet: {ip_layer.src} -> {ip_layer.dst} (Proto: {protocol_name})"

                # Add port info for TCP/UDP
                if packet.haslayer('TCP'):
                    tcp_layer = packet.getlayer('TCP')
                    log_msg += f" TCP Ports: {tcp_layer.sport} -> {tcp_layer.dport}"
                elif packet.haslayer('UDP'):
                    udp_layer = packet.getlayer('UDP')
                    log_msg += f" UDP Ports: {udp_layer.sport} -> {udp_layer.dport}"

                logging.info(log_msg)
                # print(packet.summary()) # More detailed summary

        try:
            scapy.sniff(prn=packet_handler, count=count, iface=interface, filter=filter_str, store=False)
            logging.info("Network sniffing finished.")
        except PermissionError:
            logging.error("Permission denied. Please run the script with root/administrator privileges for sniffing.")
        except OSError as e:
            logging.error(f"Network sniffing failed: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred during sniffing: {e}")


    def web_crawler(self, url: str) -> Optional[str]:
        """
        Basic web crawler/fetcher for reconnaissance. Fetches content of a single URL.

        Args:
            url (str): The URL to fetch.

        Returns:
            Optional[str]: The text content of the response, or None if an error occurs.
        """
        logging.info(f"Attempting to fetch content from: {url}")
        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout, allow_redirects=True)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            # Useful for web scraping, examining web applications
            logging.info(f"Successfully fetched {url} (Status: {response.status_code}, Size: {len(response.text)} bytes)")
            return response.text
        except requests.exceptions.Timeout:
            logging.error(f"Timeout accessing {url}")
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Error accessing {url}: {e}")
            return None


class SecurityAuditor:
    """
    Security auditing tools for web applications and services.
    Requires appropriate authorization before use against any target.
    """
    def __init__(self, user_agent: str = DEFAULT_USER_AGENT, timeout: int = DEFAULT_TIMEOUT):
        self.description = "Modular security testing framework for web services."
        self.user_agent = user_agent
        self.timeout = timeout
        self.headers = {'User-Agent': self.user_agent}

    def ssl_checker(self, domain: str) -> Dict:
        """
        Check basic SSL/TLS configuration by attempting a secure connection.
        Note: This primarily checks if the certificate is trusted by the system's
              default trust store. It does not provide detailed certificate analysis.

        Args:
            domain (str): The domain name to check (e.g., "example.com").

        Returns:
            Dict: A dictionary containing 'secure' (bool) and 'error' (str, if not secure).
        """
        url = f"https://{domain}"
        logging.info(f"Checking SSL/TLS configuration for {domain} via {url}")
        try:
            # verify=True is default, ensures cert validation
            response = requests.get(url, headers=self.headers, timeout=self.timeout, verify=True)
            response.raise_for_status()
            # If no SSLError and request succeeds, consider it basically secure from this perspective
            logging.info(f"SSL/TLS check for {domain} successful (Status: {response.status_code}).")
            return {
                "secure": True,
                "status_code": response.status_code,
                # Cannot reliably get cert info from headers
                "cert_info": "Basic check passed (Trusted Cert)"
            }
        except requests.exceptions.SSLError as e:
            logging.warning(f"SSL/TLS Error for {domain}: {e}")
            return {
                "secure": False,
                "error": f"SSLError: {e}"
            }
        except requests.exceptions.Timeout:
            logging.error(f"Timeout during SSL check for {domain}")
            return {
                "secure": False,
                "error": "Timeout during connection attempt"
            }
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed during SSL check for {domain}: {e}")
            return {
                "secure": False,
                "error": f"RequestException: {e}"
            }

    def directory_scanner(self, base_url: str, wordlist: List[str]) -> List[str]:
        """
        Scans for common directories or files on a web server using a wordlist.

        Args:
            base_url (str): The base URL to scan (e.g., "http://example.com").
            wordlist (List[str]): A list of paths/filenames to check.

        Returns:
            List[str]: A list of URLs that returned a 200 OK status.
        """
        found_dirs: List[str] = []
        # Ensure base_url ends with a slash for proper joining
        if not base_url.endswith('/'):
            base_url += '/'

        logging.info(f"Starting directory scan on {base_url} with {len(wordlist)} paths...")

        def check_path(path: str) -> None:
            path = path.strip()
            if not path: return # Skip empty lines in wordlist

            url = base_url + path # urljoin might be safer but this is common
            try:
                response = requests.get(url, headers=self.headers, timeout=self.timeout, allow_redirects=False) # Don't follow redirects usually
                # Check for 200 OK, but sometimes other 2xx or even 403 might be interesting
                if response.status_code == 200:
                    logging.info(f"Found: {url} (Status: {response.status_code})")
                    found_dirs.append(url)
                # Optionally log other statuses if needed
                # elif response.status_code == 403:
                #     logging.info(f"Potential directory (Forbidden): {url} (Status: {response.status_code})")

            except requests.exceptions.Timeout:
                logging.debug(f"Timeout checking path: {url}")
            except requests.exceptions.RequestException as e:
                logging.debug(f"Error checking path {url}: {e}")

        # Use threading for faster scanning
        with ThreadPoolExecutor(max_workers=50) as executor: # Reduced workers for web scanning
             executor.map(check_path, wordlist)

        logging.info(f"Directory scan completed for {base_url}. Found {len(found_dirs)} accessible paths.")
        return sorted(found_dirs)


def main():
    print("*" * 60)
    print("          NETWORK SECURITY TESTING & AUDITING TOOLS")
    print("*" * 60)
    print("\nWARNING: Unauthorized scanning or testing is illegal and unethical.")
    print("Ensure you have explicit permission before scanning any target.\n")

    # Get target from user input
    target = input("Enter authorized target hostname or IP address: ")
    if not target:
        print("No target specified. Exiting.")
        sys.exit(1)

    # --- Instantiate Tools ---
    tester = NetworkSecurityTester()
    auditor = SecurityAuditor()

    # --- Perform Scans ---
    print(f"\n--- Starting Scans for {target} ---")

    # 1. Port Scan (Example: Common web ports + some others)
    common_ports = list(range(1, 1025)) # Scan well-known ports
    # common_ports = [21, 22, 23, 25, 53, 80, 110, 135, 139, 143, 443, 445, 993, 995, 1723, 3306, 3389, 5900, 8080]
    print(f"\n[+] Performing Port Scan (Ports {min(common_ports)}-{max(common_ports)})...")
    open_ports = tester.port_scanner(target, common_ports)
    if open_ports:
        print(f"[+] Open Ports Found: {open_ports}")
    else:
        print("[-] No open ports found in the specified range.")

    # 2. SSL Check (only if target seems to be a domain name, not just IP)
    # Simple check if it contains a dot and no digits-only parts before the dot
    is_likely_domain = '.' in target and not all(part.isdigit() for part in target.split('.'))
    if is_likely_domain:
        print(f"\n[+] Performing SSL/TLS Check for {target}...")
        ssl_status = auditor.ssl_checker(target)
        if ssl_status.get("secure"):
            print(f"[+] SSL/TLS Status: Secure (Details: {ssl_status.get('cert_info', 'N/A')})")
        else:
            print(f"[-] SSL/TLS Status: Insecure/Error (Reason: {ssl_status.get('error', 'Unknown')})")
    else:
         # Check if port 443 is open from scan results for HTTPS check
         if 443 in open_ports:
             print(f"\n[+] Performing SSL/TLS Check for {target} (IP Address on Port 443)...")
             ssl_status = auditor.ssl_checker(target) # Use IP directly
             if ssl_status.get("secure"):
                 print(f"[+] SSL/TLS Status: Secure (Details: {ssl_status.get('cert_info', 'N/A')})")
             else:
                 print(f"[-] SSL/TLS Status: Insecure/Error (Reason: {ssl_status.get('error', 'Unknown')})")
         else:
            print(f"\n[*] Skipping SSL/TLS Check (Target '{target}' looks like an IP and port 443 is not open/scanned).")


    # 3. Directory Scan (Example - requires a wordlist file)
    wordlist_file = "common_dirs.txt" # Example wordlist filename
    try:
        # Check if port 80 or 443 is open before attempting directory scan
        web_port_open = 80 in open_ports or 443 in open_ports
        if web_port_open:
            protocol = "https" if 443 in open_ports else "http"
            base_url = f"{protocol}://{target}"
            print(f"\n[+] Performing Directory Scan on {base_url} (using '{wordlist_file}')...")
            try:
                with open(wordlist_file, 'r') as f:
                    wordlist = f.read().splitlines()

                if wordlist:
                    found_paths = auditor.directory_scanner(base_url, wordlist)
                    if found_paths:
                        print(f"[+] Found Accessible Paths:")
                        for path in found_paths:
                            print(f"  - {path}")
                    else:
                        print("[-] No accessible paths found from the wordlist.")
                else:
                    print(f"[*] Wordlist file '{wordlist_file}' is empty. Skipping directory scan.")

            except FileNotFoundError:
                print(f"[-] Wordlist file '{wordlist_file}' not found. Skipping directory scan.")
                print(f"    (Create a file named '{wordlist_file}' with one path per line, e.g., 'admin', 'login.php')")
        else:
            print("\n[*] Skipping Directory Scan (Ports 80/443 not found open in scan results).")

    except Exception as e:
        logging.error(f"An error occurred during the main execution: {e}")

    # 4. Network Sniffing (Optional - uncomment carefully)
    # print("\n[+] Performing Network Sniffing (15 packets)...")
    # print("[!] Ensure you have permissions and scapy is installed.")
    # try:
    #     tester.network_sniffer(count=15)
    # except Exception as e:
    #     print(f"[-] Sniffing failed: {e}")

    print("\n--- Security Audit Finished ---")

if __name__ == "__main__":
    main()
