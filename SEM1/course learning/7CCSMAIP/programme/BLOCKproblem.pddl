(define (problem PBLOCKS)
    (:domain BLOCKS)
    (:objects 
        A B C - block
    )
    (:init
        (ontable A)
        (ontable B)
        (on C A)
        (handempty)
        (clear B)
        (clear C)
    )
    (:goal
        (and
        (ontable C)
        (on B C)
        (on A B)
        )
    )
)