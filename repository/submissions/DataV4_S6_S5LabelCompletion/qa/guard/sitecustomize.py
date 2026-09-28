import sys
def deny(event,args):
    if event in {'socket.connect','socket.getaddrinfo'}: raise RuntimeError('OFFLINE_NETWORK_BLOCKED')
sys.addaudithook(deny)
print('OFFLINE_GUARD_ACTIVE',flush=True)
