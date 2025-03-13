#!/bin/bash
######
# Taken from https://github.com/emp-toolkit/emp-readme/blob/master/scripts/throttle.sh
######

## replace DEV=lo with your card (e.g., eth0)
#enp1s0f0
#DEV=lo 
DEV=enp1s0f0 # Tanjina-note: replaced lo with the client machine's network card
if [ "$1" == "del" ]
then
	sudo tc qdisc del dev $DEV root
fi

if [ "$1" == "lan" ]
then
sudo tc qdisc del dev $DEV root
## about 3Gbps
#sudo tc qdisc add dev $DEV root handle 1: tbf rate 3000mbit burst 100000 limit 10000
sudo tc qdisc add dev $DEV root handle 1: tbf rate 500mbit burst 100000 limit 10000 # Tanjina-note: about 500Mbps
## about 0.3ms ping latency
sudo tc qdisc add dev $DEV parent 1:1 handle 10: netem delay 0.15msec
fi
if [ "$1" == "wan" ]
then
sudo tc qdisc del dev $DEV root
## about 400Mbps
#sudo tc qdisc add dev $DEV root handle 1: tbf rate 400mbit burst 100000 limit 10000
sudo tc qdisc add dev $DEV root handle 1: tbf rate 500mbit burst 100000 limit 10000 # Tanjina-note: about 500Mbps
## about 40ms ping latency
sudo tc qdisc add dev $DEV parent 1:1 handle 10: netem delay 20msec
fi
