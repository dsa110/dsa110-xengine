from dsautils import dsa_store; d = dsa_store.DsaStore()
import sys

d.put_dict('/cmd/corr/0', {'cmd': 'trigger', 'val': sys.argv[1]})
