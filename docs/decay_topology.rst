----------------
Decay Topology
----------------

A decay chain is a simple tree, from top particle to final particles.
So the decay chain can be describing as Node (`Decay`) and Line (`Particle`)

Topology identity: The combination of final particles
-----------------------------------------------------

For example, the combination of decay chain A->RC,R->BD and A->ZB,R->CD is

`{A: [B, C, D], R: [B, D], B: [B], C: [C], D: [D]}`

and

`{A: [B, C, D], Z: [C, D], B: [B], C: [C], D: [D]}`,

The item R and Z is not same, so there are two different topology.

`{{A:[B,C,D],B:[B],C:[C],D:[D]}}` is the direct A->BCD decay.



From particles to enumerate all possible decay chain topology:
--------------------------------------------------------------
From a basic line, inserting lines to create a full graph.

from a line: `A -> B`,

insert a line (`node0 -> C`) and a node (`node0`): ::

  1. A -> node0, node0 -> B, node0 -> C

insert a line : ::

  1. A -> node0, node0 -> B, node0 -> node1, node1 -> C, node1 -> D

  2. A -> node1, node1 -> node0, node0 -> B, node0 -> C, node1 -> D

  3. A -> node0, node0 -> node1, node1 -> B, node0 -> C, node1 -> D

there are the three possible decay chains of A -> B,C,D ::

  1. A -> R+B, R -> C+D

  2. A -> R+D, R -> B+C

  3. A -> R+C, R -> B+D


the process is unique for different final particles

Each inserting process delete a line and add three new line,
So for decay process has n final particles, there are (2n-3)!! possible decay topology.
