#!/usr/bin/python
import random
import subprocess
import string
import time
import re
import os
import sys
from math import floor

#hash settings
inputSize=32
inputSizeMult=1024
nonce_size_chars=1
leading_zeroes=4
#end of hash settings

#benchmark settings
number_of_test_inputs=5000
#end of benchmark settings


if len(sys.argv)==2:
    if sys.argv[1].lower()=='--short':
        inputs=[
        '3DO73TN8U93XYEXFQDQYWONRGI45GQBY', #valid 
        '0XYUF1K6MCGYUU8295AVBKQPOZ6OD9TV', #valid 
        '5O91IDW4H9J4OAS6XL6IFKXEZZFZQDWO', #valid
        'DP9YGVG1M0OJJAK836DNTHBLHKU9DAP1', #valid
        '00000000000000000000000000000000', #invalid
        'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF', #invalid
        'BVNH5YVA9Q5MK01M9J7M9NIS5R4ZLFL6', #invalid
        '1TDNBA3AZFUEZS4HZSZ95TU99CEFL5HZ', #invalid
        ]
    else:
        print 'unknown commandline argument '+sys.argv[1]
        exit(-1)
else:
    #generate number_of_test_inputs inputs
    random.seed(1337+1)
    inputs=[''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(inputSize)) for _ in range(number_of_test_inputs)]




def getTerminalSize():
    import os
    env = os.environ
    def ioctl_GWINSZ(fd):
        try:
            import fcntl, termios, struct, os
            cr = struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ,
        '1234'))
        except:
            return
        return cr
    cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
    if not cr:
        try:
            fd = os.open(os.ctermid(), os.O_RDONLY)
            cr = ioctl_GWINSZ(fd)
            os.close(fd)
        except:
            pass
    if not cr:
        cr = (env.get('LINES', 25), env.get('COLUMNS', 80))

        ### Use get(key[, default]) instead of a try/catch
        #try:
        #    cr = (env['LINES'], env['COLUMNS'])
        #except:
        #    cr = (25, 80)
    return int(cr[1]), int(cr[0])

prev_percent=None
prev_pwidth=None
def progressbar(percentage, barName='Progress', force=False):
    #terminal size
    (width, height) = getTerminalSize()

    #progress bar width
    bar_width=width-7-len(barName)
    pwidth=int(floor(float(bar_width*float(percentage)/100.0)))

    #see if we want to redraw
    global prev_percent
    global prev_pwidth
    if prev_percent!=None and prev_pwidth!=None and force==False:
        if (prev_percent==int(percentage)) and (prev_pwidth==pwidth):
            #no need to update
            return
    prev_percent=int(percentage)
    prev_pwidth=pwidth


    sys.stdout.write('\r'+barName+' |'+('#'*pwidth).ljust(bar_width)+'|'+("%d"%(percentage)).rjust(3)+'%')
    sys.stdout.flush()

def report(msg, percentage=0.0):
    #terminal size
    (width, height) = getTerminalSize()
    
    #overwrite progress bar
    sys.stdout.write('\r'+msg.ljust(width)+'\n')

    #new progress bar
    progressbar(percentage, barName='Progress', force=False)

def check_hash(result_hash):
    if result_hash[0:leading_zeroes]=='0'*leading_zeroes:
        return True
    return False

def isValidNonce(n):
    if re.match('^[a-zA-Z0-9]+$', str(n))!=None:
        return True
    return False

start_time = time.time()
try:
    #check if we have the proper reference_hash binary
    if not os.path.isfile('./reference_hash'):
        print 'The reference_hash binary could not be found, please make sure it is in the same folder as the benchmark script'
        exit(-1)

    #fix permissions of reference hash if wrong
    if not os.access('./reference_hash', os.X_OK):
        os.system('chmod u+x ./reference_hash')

    #init score to 0
    score=0

    #start executable
    proc = subprocess.Popen(['./miner', '-benchmark'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, shell=False)

    #handle commands
    solved=0
    HashIsOut=False
    while solved!=len(inputs):
        progress=float(solved)/float(len(inputs))*100.0
        progressbar(progress)

        #read input from process
        cmd=[s.strip() for s in proc.stdout.readline().split() if s.strip()!='']
        if len(cmd)==0 or len(cmd)>3:
            report('Your program provided an unsupported command', progress)
            proc.stdin.write('NAK\n')
            continue

        #R              //request new input
        if cmd[0]=='R': #new input
            if HashIsOut:
                report('Your program is trying to obtain new hash before solving previous...this is not allowed!', progress)
                proc.stdin.write("NAK\n")
                continue
           
            hashOut=inputs[solved]
            proc.stdin.write(hashOut+'\n')
            HashIsOut=True
        
        #V hash nonce   //validate hash + nonce ('NONE' if no valid hash was found)
        elif cmd[0]=='V': #validate hash
            if not HashIsOut:
                report('Trying to solve hash but none was requested!', progress)
                proc.stdin.write('NAK\n')
                continue
            HashIsOut=False
            if len(cmd)!=3:
                report("'V' command given with incorrect number of parameters!", progress)
                proc.stdin.write('NAK\n')
                continue
            hash=str(cmd[1])
            if hash!=hashOut:
                report('Trying to validate different hash than what was supplied!', progress)
                proc.stdin.write('NAK\n')
                continue

            nonce=str(cmd[2]).strip()
            
            if nonce.lower()=='none':
                #print'Your program claimed there are no solutions to given hash (this can be normal behavior if no hash is found!)'
                solved+=1
                proc.stdin.write('ACK\n')
                continue

            if len(nonce)!=nonce_size_chars:
                report('Provided Nonce is too large!',progress)
                proc.stdin.write('NAK\n')
                continue

            if not isValidNonce(nonce):
                report('Illegal Nonce provided!',progress)
                proc.stdin.write('NAK\n')
                continue

            else:
                #check hash
                hash_proc = subprocess.Popen(['./reference_hash', str(hash), str(inputSizeMult), str(nonce)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                result_hash= hash_proc.stdout.readline().strip()
                report(' '.join([str(hash), str(inputSizeMult), str(nonce), str(result_hash)]), progress)
                if check_hash(result_hash):
                    solved+=1
                    score+=1
                    proc.stdin.write('ACK\n')
                    report('Congratulations, you found a valid coin!', progress)
                else:
                    report('Your program supplied an invalid solution!', progress)
                    proc.stdin.write('NAK\n')
                    # round_invalid+=1
        else:
            report('Your program passed an unsupported command', progress)
            proc.stdin.write('NAK\n')
    
    #all test inputs are done, terminate program
    proc.terminate()
except:
    print'The program closed unexpectedly...'
    exit(0)

#
progressbar(100.0)
print 'Processed all inputs'
print '-'*40
print 'Summary:'
print '-'*40
print "%-20s %d"%('- Processed Blocks:',solved)
print "%-20s %d"%('- Number of Coins:', score)
print "%-20s %d sec"%('- Running Time:', time.time() - start_time)
print '-'*40

