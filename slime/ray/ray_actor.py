from slime.utils.misc import get_current_node_ip, get_free_port


class RayActor:
    @staticmethod
    def _get_current_node_ip_and_free_port(start_port=10000, consecutive=1):
        return get_current_node_ip(), get_free_port(start_port=start_port, consecutive=consecutive)

    @staticmethod
    def _get_current_node_ip_and_free_ports(start_port=10000, consecutives=None):
        node_ip = get_current_node_ip()
        ports = []
        next_start_port = start_port

        for consecutive in consecutives or []:
            port = get_free_port(start_port=next_start_port, consecutive=consecutive)
            ports.append(port)
            next_start_port = port + consecutive

        return node_ip, ports, next_start_port

    def get_master_addr_and_port(self):
        return self.master_addr, self.master_port
