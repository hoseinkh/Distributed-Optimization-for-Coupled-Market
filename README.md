# Distributed-Optimization-for-Coupled-Market

This project will implement an intra-market between several regional Day-Ahead electricity markets.
Solving the intra-market is in essence solving a distributed optimization.
The members of the intra-market are the operators of several regional markets that participate in it.
The goal is that each market share **very little information** about the underlying private 
information in their regional markets, and yet the intra-market achieves **efficiency**.

The intra-day market is an optimization problem that the players (operators of the regional markets) 
submit their bids, a security-constraint optimization problem is solved.

The proposed market is based on an iterative mechanism where the operators of the regional markets that are 
coupled through tie-lines share limited amount of information, update their bids, and repeat until the 
interative algorithm converges.

The goal is that, if an _omniscient_ agent that knows all the private information about the regional 
markets solve this problem without a intermdediate tool (i.e. intra-market), we would get the same 
result as the one we get from implementing this mechanism.

**Details on the code**

The distributed optimization, the intra-market, and the regional markets are implemented in a class called 
**_ClassHmatpower_**. This class is compatible with the package [MATPOWER](https://matpower.org/) for further investigations.

A 14 bus case with three regional markets is implemented.

For more details on this project, see [link](https://matpower.org/).

