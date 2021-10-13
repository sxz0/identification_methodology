
# GPU performance tests extracted from py-videocorevi Python library.
# Testing for Raspberry Pi 4 Benchmarking and device identification.
# TREASURE PROJECT 2021


import time
from time import clock_gettime,CLOCK_MONOTONIC
from time import monotonic
import fcntl
import socket
import struct
import numpy as np
from videocore6.v3d import *
from videocore6 import pack_unpack
from videocore6.driver import Driver
from videocore6.assembler import qpu
from bench_helper import BenchHelper
import sys
import os
import random
import hashlib

def getsec():
    return clock_gettime(CLOCK_MONOTONIC)

@qpu
def load_params(asm, thread, regs):

    if thread == 1:
        bxor(r0, r0, r0, sig = ldunifrf(rf0))
    elif thread == 8:
        #  8 threads (1 threads / qpu)
        tidx(r0, sig = ldunifrf(rf0))
        shr(r0, r0, 2)
        mov(r1, 0b1111)
    elif thread == 16:
        # 16 threads (2 threads / qpu)
        tidx(r0, sig = ldunifrf(rf0))
        shr(r0, r0, 1).mov(r1, 1)
        shl(r1, r1, 5)
        sub(r1, r1, 1)
    else:
        assert thread in [1,8,16]

    band(r3, r0, r1, sig = ldunifrf(rf1))
    shl(r0, rf1, 2)
    umul24(r0, r0, r3)
    eidx(r1).add(r0, r0, rf0)
    shl(r1, r1, 2)
    shl(r3, 4, 4).add(r0, r0, r1)
    n = len(regs)
    mov(tmua, r0, sig = thrsw).add(r0, r0, r3)
    nop()
    nop()
    nop(sig = ldtmu(r1))
    for i in range(n):
        if i % 16 == 0:
            mov(r5rep, r1)
            mov(regs[i], r5)
        elif i % 16 == 15 and i != n - 1:
            mov(tmua, r0, sig = thrsw).add(r0, r0, r3)
            rotate(r5rep, r1, - (i % 16))
            mov(regs[i], r5)
            nop(sig = ldtmu(r1))
        else:
            rotate(r5rep, r1, - (i % 16))
            mov(regs[i], r5)

@qpu
def qpu_sgemm_rnn_naive(asm, thread):

    params = [
        'P',
        'Q',
        'R',
        'A_base',
        'A_stride',
        'B_base',
        'B_stride',
        'C_base',
        'C_stride',
        'alpha',
        'beta',
    ]

    values = [
        'A_cur',
        'B_cur',
        'C_cur',
        'i', 'j', 'k',
    ]

    g = globals()
    for i, reg in enumerate(params + values):
        g['reg_' + reg] = g['rf' + str(i+32)]

    load_params(asm, thread, [g['reg_' + reg] for reg in params])

    add(r0, reg_P, 15)
    shr(r0, r0, 4)
    shl(r0, r0, 4)
    add(r1, reg_R, 15)
    shr(r1, r1, 4)
    shl(r1, r1, 6)
    umul24(r3, r0, reg_A_stride)
    add(reg_A_base, reg_A_base, r3)
    add(reg_B_base, reg_B_base, r1)
    umul24(r3, r0, reg_C_stride)
    add(reg_C_base, reg_C_base, r3)
    add(reg_C_base, reg_C_base, r1)

    for i in range(16):
        mov(rf[i], 0.0).mov(rf[i+16], 0.0)

    # i=(p+15)/16.
    add(r0, reg_P, 15)
    shr(reg_i, r0, 4)
    with loop as li:

        # j=(r+15)/16
        add(r0, reg_R, 15)
        shr(reg_j, r0, 4)
        with loop as lj:

            shl(r0, reg_i, 4)
            umul24(r3, r0, reg_C_stride)
            shl(r1, reg_j, 6)
            sub(reg_C_cur, reg_C_base, r3)
            sub(reg_C_cur, reg_C_cur, r1)
            umul24(r3, r0, reg_A_stride)
            sub(reg_A_cur, reg_A_base, r3)
            sub(reg_B_cur, reg_B_base, r1)

            mov(reg_k, reg_Q)
            with loop as lk:

                eidx(r0)
                umul24(r1, r0, reg_A_stride)
                add(r1, r1, reg_A_cur).add(reg_A_cur, reg_A_cur, 4)
                mov(tmua, r1, sig = thrsw)
                shl(r1, r0, 2)
                add(r1, r1, reg_B_cur).add(reg_B_cur, reg_B_cur, reg_B_stride)
                mov(tmua, r1, sig = thrsw)

                nop(sig = ldtmu(r0))
                mov(r5rep, r0)
                nop(sig = ldtmu(r4))
                nop().fmul(r3, r5, r4)
                for i in range(1,16):
                    rotate(r5rep, r0, -i)
                    fadd(rf[i-1], rf[i-1], r3).fmul(r3, r5, r4)
                fadd(rf15, rf15, r3)

                sub(reg_k, reg_k, 1, cond = 'pushz')
                lk.b(cond = 'anyna')
                nop() # delay slot
                nop() # delay slot
                nop() # delay slot

            eidx(r0)
            shl(r0, r0, 2)
            add(r1, reg_C_cur, r0)
            mov(tmua, r1, sig = thrsw).add(r1, r1, reg_C_stride)
            fmul(rf[0], rf[0], reg_alpha)
            for i in range(1, 16):
                mov(tmua, r1, sig = thrsw).add(r1, r1, reg_C_stride)
                fmul(rf[i], rf[i], reg_alpha, sig = ldtmu(rf[i+15]))
            mov(r0, reg_beta).fmul(r3, rf[16], reg_beta, sig = ldtmu(rf[31]))
            for i in range(16):
                fadd(rf[i], rf[i], r3).fmul(r3, rf[i+17], r0)

            eidx(r0)
            shl(r0, r0, 2)
            add(r1, reg_C_cur, r0)
            for i in range(16):
                mov(tmud, rf[i])
                mov(tmua, r1).add(r1, r1, reg_C_stride)
                mov(rf[i], 0.0).mov(rf[i+16], 0.0)
                tmuwt()

            sub(reg_j, reg_j, 1, cond = 'pushz')
            lj.b(cond = 'anyna')
            nop() # delay slot
            nop() # delay slot
            nop() # delay slot

        sub(reg_i, reg_i, 1, cond = 'pushz')
        li.b(cond = 'anyna')
        nop()
        nop()
        nop()

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def sgemm_rnn_naive():

    thread = 8

    P = 1024
    Q = 1024
    R = 1024

    assert P % (16 * 2) == 0
    assert R % (16 * 4) == 0

    with Driver() as drv:

        code = drv.program(lambda asm: qpu_sgemm_rnn_naive(asm, thread))

        A = drv.alloc((P, Q), dtype = 'float32')
        B = drv.alloc((Q, R), dtype = 'float32')
        C = drv.alloc((P, R), dtype = 'float32')

        np.random.seed(0)
        alpha = np.random.randn()
        beta = np.random.randn()
        A_ref = np.random.randn(*A.shape).astype(A.dtype)
        B_ref = np.random.randn(*B.shape).astype(B.dtype)
        C_ref = np.random.randn(*C.shape).astype(C.dtype)

        A[:] = A_ref
        B[:] = B_ref
        C[:] = C_ref

        start = time.perf_counter_ns()
        C_ref[:] = alpha * A_ref.dot(B_ref) + beta * C_ref
        time_ref = time.perf_counter_ns() - start

        def block_2x4_params(i, j):
            tile_P = P // 2
            tile_R = R // 4
            return [
                tile_P, Q, tile_R,
                A.addresses()[tile_P*i, 0       ],
                A.strides[0],
                B.addresses()[0       , tile_R*j],
                B.strides[0],
                C.addresses()[tile_P*i, tile_R*j],
                C.strides[0],
                *pack_unpack('f', 'I', [alpha, beta]),
            ]

        unif_params = drv.alloc((thread, len(block_2x4_params(0,0))), dtype = 'uint32')
        for th in range(thread):
            unif_params[th] = block_2x4_params(th // 4, th % 4)

        unif = drv.alloc(2, dtype = 'uint32')
        unif[0] = unif_params.addresses()[0,0]
        unif[1] = unif_params.shape[1]

        start = time.perf_counter_ns()
        drv.execute(code, unif.addresses()[0], thread = thread)
        time_gpu = time.perf_counter_ns() - start

        np.set_printoptions(threshold=np.inf)

        def Gflops(sec):
            return (2 * P * Q * R + 3 * P * R) / sec * 1e-9

        return [time_ref,time_gpu] #Gflops(time_ref),time_gpu,Gflops(time_gpu)]

def sleep(duration):
    duration=duration*1000000000
    now = time.perf_counter_ns()
    end = now + duration
    while now < end:
        now = time.perf_counter_ns()

def get_QPU_freq(seg):
	with RegisterMapping() as regmap:
		with PerformanceCounter(regmap, [CORE_PCTR_CYCLE_COUNT]) as pctr:
			time.sleep(seg)
			result = pctr.result()
			return (result[0] * 1e-6)

def cpu_random():
        with RegisterMapping() as regmap:
                with PerformanceCounter(regmap, [CORE_PCTR_CYCLE_COUNT]) as pctr:
                        a=random.random()
                        result = pctr.result()
                        return (result[0])

def cpu_true_random(n):
        with RegisterMapping() as regmap:
                with PerformanceCounter(regmap, [CORE_PCTR_CYCLE_COUNT]) as pctr:
                        a=os.urandom(n)
                        result = pctr.result()
                        return (result[0])

def cpu_hash():
	with RegisterMapping() as regmap:
		with PerformanceCounter(regmap, [CORE_PCTR_CYCLE_COUNT]) as pctr:
			h=int(hashlib.sha256("test string".encode('utf-8')).hexdigest(), 16) % 10**8
			result = pctr.result()
			return (result[0])

@qpu
def qpu_summation(asm, *, num_qpus, unroll_shift, code_offset,
                  align_cond=lambda pos: pos % 512 == 170):

    g = globals()
    for i, v in enumerate(['length', 'src', 'dst', 'qpu_num', 'stride', 'sum']):
        g[f'reg_{v}'] = rf[i]

    nop(sig=ldunifrf(reg_length))
    nop(sig=ldunifrf(reg_src))
    nop(sig=ldunifrf(reg_dst))

    if num_qpus == 1:
        num_qpus_shift = 0
        mov(reg_qpu_num, 0)
    elif num_qpus == 8:
        num_qpus_shift = 3
        tidx(r0)
        shr(r0, r0, 2)
        band(reg_qpu_num, r0, 0b1111)
    else:
        raise Exception('num_qpus must be 1 or 8')

    # addr += 4 * (thread_num + 16 * qpu_num)
    shl(r0, reg_qpu_num, 4)
    eidx(r1)
    add(r0, r0, r1)
    shl(r0, r0, 2)
    add(reg_src, reg_src, r0).add(reg_dst, reg_dst, r0)

    # stride = 4 * 16 * num_qpus
    mov(reg_stride, 1)
    shl(reg_stride, reg_stride, 6 + num_qpus_shift)

    # The QPU performs shifts and rotates modulo 32, so it actually supports
    # shift amounts [0, 31] only with small immediates.
    num_shifts = [*range(16), *range(-16, 0)]

    # length /= 16 * 8 * num_qpus * unroll
    shr(reg_length, reg_length, num_shifts[7 + num_qpus_shift + unroll_shift])

    # This single thread switch and two instructions just before the loop are
    # really important for TMU read to achieve a better performance.
    # This also enables TMU read requests without the thread switch signal, and
    # the eight-depth TMU read request queue.
    nop(sig=thrsw)
    nop()
    bxor(reg_sum, 1, 1).mov(r1, 1)

    while not align_cond(code_offset + len(asm)):
        nop()

    with loop as l:

        unroll = 1 << unroll_shift

        for i in range(7):
            mov(tmua, reg_src).add(reg_src, reg_src, reg_stride)
        mov(tmua, reg_src).sub(reg_length, reg_length, r1, cond='pushz')
        add(reg_src, reg_src, reg_stride, sig=ldtmu(r0))

        for j in range(unroll - 1):
            for i in range(8):
                mov(tmua, reg_src).add(reg_src, reg_src, reg_stride)
                add(reg_sum, reg_sum, r0, sig=ldtmu(r0))

        for i in range(5):
            add(reg_sum, reg_sum, r0, sig=ldtmu(r0))

        l.b(cond='na0')
        add(reg_sum, reg_sum, r0, sig=ldtmu(r0))  # delay slot
        add(reg_sum, reg_sum, r0, sig=ldtmu(r0))  # delay slot
        add(reg_sum, reg_sum, r0)                 # delay slot

    mov(tmud, reg_sum)
    mov(tmua, reg_dst)

    # This synchronization is needed between the last TMU operation and the
    # program end with the thread switch just before the loop above.
    barrierid(syncb, sig=thrsw)
    nop()
    nop()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()

def summation(*, length, num_qpus=8, unroll_shift=5):

    assert length > 0
    assert length % (16 * 8 * num_qpus * (1 << unroll_shift)) == 0

    with Driver(data_area_size=(length + 1024) * 4) as drv:

        code = drv.program(qpu_summation, num_qpus=num_qpus,
                           unroll_shift=unroll_shift,
                           code_offset=drv.code_pos // 8)

        X = drv.alloc(length, dtype='uint32')
        Y = drv.alloc(16 * num_qpus, dtype='uint32')

        X[:] = np.arange(length, dtype=X.dtype)
        Y.fill(0)

        assert sum(Y) == 0

        unif = drv.alloc(3, dtype='uint32')
        unif[0] = length
        unif[1] = X.addresses()[0]
        unif[2] = Y.addresses()[0]


        start = time.perf_counter_ns()
        drv.execute(code, unif.addresses()[0], thread=num_qpus)
        end = time.perf_counter_ns()

        assert sum(Y) % 2**32 == (length - 1) * length // 2 % 2**32
        return [end - start] #,length * 4 / (end - start) * 1e-6]

@qpu
def qpu_scopy(asm, *, num_qpus, unroll_shift, code_offset,
              align_cond=lambda pos: pos % 512 == 259):

    g = globals()
    for i, v in enumerate(['length', 'src', 'dst', 'qpu_num', 'stride']):
        g[f'reg_{v}'] = rf[i]

    nop(sig=ldunifrf(reg_length))
    nop(sig=ldunifrf(reg_src))
    nop(sig=ldunifrf(reg_dst))

    if num_qpus == 1:
        num_qpus_shift = 0
        mov(reg_qpu_num, 0)
    elif num_qpus == 8:
        num_qpus_shift = 3
        tidx(r0)
        shr(r0, r0, 2)
        band(reg_qpu_num, r0, 0b1111)
    else:
        raise Exception('num_qpus must be 1 or 8')

    # addr += 4 * (thread_num + 16 * qpu_num)
    shl(r0, reg_qpu_num, 4)
    eidx(r1)
    add(r0, r0, r1)
    shl(r0, r0, 2)
    add(reg_src, reg_src, r0).add(reg_dst, reg_dst, r0)

    # stride = 4 * 16 * num_qpus
    mov(reg_stride, 1)
    shl(reg_stride, reg_stride, 6 + num_qpus_shift)

    # length /= 16 * 8 * num_qpus * unroll
    shr(reg_length, reg_length, 7 + num_qpus_shift + unroll_shift)

    # This single thread switch and two nops just before the loop are really
    # important for TMU read to achieve a better performance.
    # This also enables TMU read requests without the thread switch signal, and
    # the eight-depth TMU read request queue.
    nop(sig=thrsw)
    nop()
    nop()

    while not align_cond(code_offset + len(asm)):
        nop()

    with loop as l:

        unroll = 1 << unroll_shift

        for i in range(8):
            mov(tmua, reg_src).add(reg_src, reg_src, reg_stride)

        for j in range(unroll - 1):
            for i in range(8):
                nop(sig=ldtmu(r0))
                mov(tmua, reg_src).add(reg_src, reg_src, reg_stride)
                mov(tmud, r0)
                mov(tmua, reg_dst).add(reg_dst, reg_dst, reg_stride)

        for i in range(6):
            nop(sig=ldtmu(r0))
            mov(tmud, r0)
            mov(tmua, reg_dst).add(reg_dst, reg_dst, reg_stride)

        nop(sig=ldtmu(r0))
        mov(tmud, r0).sub(reg_length, reg_length, 1, cond='pushz')
        mov(tmua, reg_dst).add(reg_dst, reg_dst, reg_stride)

        l.b(cond='na0')
        nop(sig=ldtmu(r0))                                    # delay slot
        mov(tmud, r0)                                         # delay slot
        mov(tmua, reg_dst).add(reg_dst, reg_dst, reg_stride)  # delay slot

    # This synchronization is needed between the last TMU operation and the
    # program end with the thread switch just before the loop above.
    barrierid(syncb, sig=thrsw)
    nop()
    nop()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()

def scopy(*, length, num_qpus=8, unroll_shift=0):

    assert length > 0
    assert length % (16 * 8 * num_qpus * (1 << unroll_shift)) == 0

    with Driver(data_area_size=(length * 2 + 1024) * 4) as drv:

        code = drv.program(qpu_scopy, num_qpus=num_qpus,
                           unroll_shift=unroll_shift,
                           code_offset=drv.code_pos // 8)

        X = drv.alloc(length, dtype='float32')
        Y = drv.alloc(length, dtype='float32')

        X[:] = np.arange(*X.shape, dtype=X.dtype)
        Y[:] = -X

        assert not np.array_equal(X, Y)

        unif = drv.alloc(3, dtype='uint32')
        unif[0] = length
        unif[1] = X.addresses()[0]
        unif[2] = Y.addresses()[0]

        start = time.perf_counter_ns()
        drv.execute(code, unif.addresses()[0], thread=num_qpus)
        end = time.perf_counter_ns()

        assert np.array_equal(X, Y)

        return[end - start] #, length * 4 / (end - start) * 1e-6]

@qpu
def qpu_memset(asm, *, num_qpus, unroll_shift, code_offset,
               align_cond=lambda pos: pos % 512 == 0):

    g = globals()
    for i, v in enumerate(['dst', 'fill', 'length', 'qpu_num', 'stride']):
        g[f'reg_{v}'] = rf[i]

    nop(sig=ldunifrf(reg_dst))
    nop(sig=ldunifrf(reg_fill))
    nop(sig=ldunifrf(reg_length))

    if num_qpus == 1:
        num_qpus_shift = 0
        mov(reg_qpu_num, 0)
    elif num_qpus == 8:
        num_qpus_shift = 3
        tidx(r0)
        shr(r0, r0, 2)
        band(reg_qpu_num, r0, 0b1111)
    else:
        raise Exception('num_qpus must be 1 or 8')

    # addr += 4 * (thread_num + 16 * qpu_num)
    shl(r0, reg_qpu_num, 4)
    eidx(r1)
    add(r0, r0, r1)
    shl(r0, r0, 2)
    add(reg_dst, reg_dst, r0)

    # stride = 4 * 16 * num_qpus
    # r0 = 1
    mov(r0, 1)
    shl(reg_stride, r0, 6 + num_qpus_shift)

    # length /= 16 * num_qpus * unroll
    shr(reg_length, reg_length, 4 + num_qpus_shift + unroll_shift)

    unroll = 1 << unroll_shift

    if unroll == 1:

        sub(reg_length, reg_length, r0, cond='pushz')

        while not align_cond(code_offset + len(asm)):
            nop()

        with loop as l:

            l.b(cond='na0')
            mov(tmud, reg_fill)                                   # delay slot
            mov(tmua, reg_dst).add(reg_dst, reg_dst, reg_stride)  # delay slot
            sub(reg_length, reg_length, r0, cond='pushz')         # delay slot

    else:

        while not align_cond(code_offset + len(asm)):
            nop()

        with loop as l:

            for i in range(unroll - 2):
                mov(tmud, reg_fill)
                mov(tmua, reg_dst).add(reg_dst, reg_dst, reg_stride)

            mov(tmud, reg_fill).sub(reg_length, reg_length, r0, cond='pushz')
            l.b(cond='na0')
            mov(tmua, reg_dst).add(reg_dst, reg_dst, reg_stride)  # delay slot
            mov(tmud, reg_fill)                                   # delay slot
            mov(tmua, reg_dst).add(reg_dst, reg_dst, reg_stride)  # delay slot

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def memset(*, fill, length, num_qpus=8, unroll_shift=1):

    assert length > 0
    assert length % (16 * num_qpus * (1 << unroll_shift)) == 0

    with Driver(data_area_size=(length + 1024) * 4) as drv:

        code = drv.program(qpu_memset, num_qpus=num_qpus,
                           unroll_shift=unroll_shift,
                           code_offset=drv.code_pos // 8)

        X = drv.alloc(length, dtype='uint32')

        X.fill(~fill)

        assert not np.array_equiv(X, fill)

        unif = drv.alloc(3, dtype='uint32')
        unif[0] = X.addresses()[0]
        unif[1] = fill
        unif[2] = length

        start = monotonic()
        drv.execute(code, unif.addresses()[0], thread=num_qpus)
        end = monotonic()

        assert np.array_equiv(X, fill)

        return [end - start] #, length * 4 / (end - start) * 1e-6]


@qpu
def qpu_clock(asm):

    nop(sig = ldunif)
    nop(sig = ldunifrf(rf0))

    with loop as l:
        sub(r5, r5, 1, cond = 'pushn')
        l.b(cond = 'anyna')
        nop()
        nop()
        nop()

    mov(tmud, 1)
    mov(tmua, rf0)
    tmuwt()

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()


def test_clock():

    bench = BenchHelper('./libbench_helper.so')

    with Driver() as drv:

        f = pow(2, 25)

        code = drv.program(qpu_clock)
        unif = drv.alloc(2, dtype = 'uint32')
        done = drv.alloc(1, dtype = 'uint32')

        done[:] = 0

        unif[0] = f
        unif[1] = done.addresses()[0]

        with drv.compute_shader_dispatcher() as csd:
            start = time.perf_counter_ns()
            csd.dispatch(code, unif.addresses()[0])
            bench.wait_address(done)
            end = time.perf_counter_ns()
            return [f * 5 / (end - start) / 1000 / 1000 * 4] #end - start] #, f * 5 / (end - start) / 1000 / 1000 * 4]


@qpu
def qpu_write_N(asm, N):

    eidx(r0, sig = ldunif)
    nop(sig = ldunifrf(rf0))
    shl(r0, r0, 2)
    mov(tmud, N)
    add(tmua, r5, r0)
    tmuwt()

    mov(tmud, 1)
    mov(tmua, rf0)
    tmuwt()

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def test_multiple_dispatch_delay():

    bench = BenchHelper('./libbench_helper.so')

    with Driver() as drv:

        data = drv.alloc((5, 16), dtype = 'uint32')
        code = [drv.program(lambda asm: qpu_write_N(asm, i)) for i in range(data.shape[0])]
        unif = drv.alloc((data.shape[0], 2), dtype = 'uint32')
        done = drv.alloc(1, dtype = 'uint32')

        data[:] = 0
        unif[:,0] = data.addresses()[:,0]
        unif[:,1] = done.addresses()[0]

        ref_start = time.perf_counter_ns()
        with drv.compute_shader_dispatcher() as csd:
            for i in range(data.shape[0]):
                csd.dispatch(code[i], unif.addresses()[i,0])
        ref_end = time.perf_counter_ns()
        assert (data == np.arange(data.shape[0]).reshape(data.shape[0],1)).all()

        data[:] = 0

        naive_results = np.zeros(data.shape[0], dtype='float32')
        with drv.compute_shader_dispatcher() as csd:
            for i in range(data.shape[0]):
                done[:] = 0
                start = time.perf_counter_ns()
                csd.dispatch(code[i], unif.addresses()[i,0])
                bench.wait_address(done)
                end = time.perf_counter_ns()
                naive_results[i] = end - start
        assert (data == np.arange(data.shape[0]).reshape(data.shape[0],1)).all()

        sleep_results = np.zeros(data.shape[0], dtype='float32')
        with drv.compute_shader_dispatcher() as csd:
            for i in range(data.shape[0]):
                done[:] = 0
                time.sleep(1)
                start = time.perf_counter_ns()
                csd.dispatch(code[i], unif.addresses()[i,0])
                bench.wait_address(done)
                end = time.perf_counter_ns()
                sleep_results[i] = end - start
        assert (data == np.arange(data.shape[0]).reshape(data.shape[0],1)).all()
        return [ref_end - ref_start,np.sum(naive_results),np.sum(sleep_results)]

@qpu
def qpu_tmu_load_1_slot_1_qpu(asm, nops):

    nop(sig = ldunifrf(rf0)) # X.shape[1]
    nop(sig = ldunifrf(rf1)) # X
    nop(sig = ldunifrf(rf2)) # X.stride[1]
    nop(sig = ldunifrf(rf3)) # X.stride[0]
    nop(sig = ldunifrf(rf4)) # Y
    nop(sig = ldunifrf(rf5)) # done

    barrierid(syncb, sig = thrsw)
    nop()
    nop()

    tidx(r0)
    shr(r0, r0, 2)
    band(r0, r0, 0b1111, cond = 'pushz')
    b(R.done, cond = 'allna')
    nop() # delay slot
    nop() # delay slot
    nop() # delay slot

    eidx(r0)
    shl(r0, r0, 2)
    add(rf4, rf4, r0)

    eidx(r0)
    umul24(r0, r0, rf3)
    add(rf1, rf1, r0)

    mov(r2, 0.0)
    with loop as l:
        mov(tmua, rf1).add(rf1, rf1, rf2)
        for i in range(nops):
            nop()
        nop(sig = ldtmu(r3))
        sub(rf0, rf0, 1, cond = 'pushz')
        l.b(cond = 'anyna')
        fadd(r2, r2, r3) # delay slot
        nop()            # delay slot
        nop()            # delay slot

    mov(tmud, r2)
    mov(tmua, rf4)
    tmuwt()

    mov(tmud, 1)
    mov(tmua, rf5)
    tmuwt()

    L.done
    barrierid(syncb, sig = thrsw)
    nop()
    nop()

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def test_tmu_load_1_slot_1_qpu():

    bench = BenchHelper('./libbench_helper.so')
    res = []
    for trans in [False, True]:

        with Driver() as drv:

            loop = 2**15

            X = drv.alloc((16, loop) if trans else (loop, 16), dtype = 'float32')
            Y = drv.alloc(16, dtype = 'float32')
            unif = drv.alloc(6, dtype = 'uint32')
            done = drv.alloc(1, dtype = 'uint32')

            unif[0] = loop
            unif[1] = X.addresses()[0,0]
            unif[2] = X.strides[int(trans)]
            unif[3] = X.strides[1-int(trans)]
            unif[4] = Y.addresses()[0]
            unif[5] = done.addresses()[0]

            results = np.zeros((1, 10), dtype = 'float32')

            #fig = plt.figure()
            #ax = fig.add_subplot(1,1,1)
            #ax.set_title(f'TMU load latency (1 slot, 1 qpu, stride=({unif[2]},{unif[3]}))')
            #ax.set_xlabel('# of nop (between request and load signal)')
            #ax.set_ylabel('sec')

            for nops in range(results.shape[0]):

                code = drv.program(lambda asm: qpu_tmu_load_1_slot_1_qpu(asm, nops))

                for i in range(results.shape[1]):

                    with drv.compute_shader_dispatcher() as csd:

                        X[:] = np.random.randn(*X.shape) / X.shape[int(trans)]
                        Y[:] = 0.0
                        done[:] = 0

                        start = time.perf_counter_ns()
                        csd.dispatch(code, unif.addresses()[0], thread = 8)
                        bench.wait_address(done)
                        end = time.perf_counter_ns()

                        results[nops,i] = end - start

                        assert np.allclose(Y, np.sum(X, axis=int(trans)), atol = 1e-4)

                #ax.scatter(np.zeros(results.shape[1])+nops, results[nops], s=1, c='blue')

                #print('{:4}/{}\t{:.9f}'.format(nops, results.shape[0], np.sum(results[nops]) / results.shape[1]))
                res.append(np.sum(results[nops]) / results.shape[1])
    return res
            #ax.set_ylim(auto=True)
            #ax.set_xlim(0, results.shape[0])
            #fig.savefig(f'benchmarks/tmu_load_1_slot_1_qpu_{unif[2]}_{unif[3]}.png')

@qpu
def qpu_tmu_load_2_slot_1_qpu(asm, nops):

    nop(sig = ldunifrf(rf0)) # X.shape[1]
    nop(sig = ldunifrf(rf1)) # X
    nop(sig = ldunifrf(rf2)) # X.stride[1]
    nop(sig = ldunifrf(rf3)) # X.stride[0]
    nop(sig = ldunifrf(rf4)) # Y
    nop(sig = ldunifrf(rf5)) # done

    barrierid(syncb, sig = thrsw)
    nop()
    nop()

    tidx(r0)
    shr(r0, r0, 2)
    band(r0, r0, 0b0011, cond = 'pushz')
    b(R.skip_bench, cond = 'allna')
    nop()
    nop()
    nop()

    eidx(r0)
    shl(r0, r0, 2)
    add(rf4, rf4, r0)
    tidx(r0)
    shr(r0, r0, 2)
    band(r0, r0, 0b1111)
    shl(r1, 4, 4)
    umul24(r0, r0, r1)
    add(rf4, rf4, r0)

    eidx(r0)
    umul24(r0, r0, rf3)
    add(rf1, rf1, r0)
    tidx(r0)
    shr(r0, r0, 2)
    band(r0, r0, 0b1111)
    shl(r1, rf0, 6)
    umul24(r0, r0, r1)
    add(rf1, rf1, r0)

    mov(r2, 0.0)
    with loop as l:
        mov(tmua, rf1).add(rf1, rf1, rf2)
        for i in range(nops):
            nop()
        nop(sig = ldtmu(r3))
        sub(rf0, rf0, 1, cond = 'pushz')
        l.b(cond = 'anyna')
        fadd(r2, r2, r3) # delay slot
        nop()            # delay slot
        nop()            # delay slot

    mov(tmud, r2)
    mov(tmua, rf4)
    tmuwt()

    L.skip_bench

    barrierid(syncb, sig = thrsw)
    nop()
    nop()

    tidx(r0)
    shr(r0, r0, 2)
    band(r0, r0, 0b1111, cond = 'pushz')
    b(R.skip_done, cond = 'allna')
    nop()
    nop()
    nop()
    mov(tmud, 1)
    mov(tmua, rf5)
    tmuwt()
    L.skip_done

    nop(sig = thrsw)
    nop(sig = thrsw)
    nop()
    nop()
    nop(sig = thrsw)
    nop()
    nop()
    nop()

def test_tmu_load_2_slot_1_qpu():

    bench = BenchHelper('./libbench_helper.so')
    res=[]
    for trans, min_nops, max_nops in [(False, 0, 1), (True, 0, 1)]:

        with Driver() as drv:

            loop = 2**13

            X = drv.alloc((8, 16, loop) if trans else (8, loop, 16), dtype = 'float32')
            Y = drv.alloc((8, 16), dtype = 'float32')
            unif = drv.alloc(6, dtype = 'uint32')
            done = drv.alloc(1, dtype = 'uint32')

            unif[0] = loop
            unif[1] = X.addresses()[0,0,0]
            unif[2] = X.strides[1+int(trans)]
            unif[3] = X.strides[2-int(trans)]
            unif[4] = Y.addresses()[0,0]
            unif[5] = done.addresses()[0]

            results = np.zeros((max_nops, 10), dtype = 'float32')

            #fig = plt.figure()
            #ax = fig.add_subplot(1,1,1)
            #ax.set_title(f'TMU load latency (2 slot, 1 qpu, stride=({unif[2]},{unif[3]}))')
            #ax.set_xlabel('# of nop (between request and load signal)')
            #ax.set_ylabel('sec')

            #print()
            for nops in range(min_nops, results.shape[0]):

                code = drv.program(lambda asm: qpu_tmu_load_2_slot_1_qpu(asm, nops))

                for i in range(results.shape[1]):

                    with drv.compute_shader_dispatcher() as csd:

                        X[:] = np.random.randn(*X.shape) / X.shape[1+int(trans)]
                        Y[:] = 0.0
                        done[:] = 0

                        start = time.perf_counter_ns()
                        csd.dispatch(code, unif.addresses()[0], thread = 8)
                        bench.wait_address(done)
                        end = time.perf_counter_ns()

                        results[nops,i] = end - start

                        assert np.allclose(Y[0::4], np.sum(X[0::4], axis=1+int(trans)), atol = 1e-4)
                        assert (Y[1:4] == 0).all()
                        assert (Y[5:8] == 0).all()

                #ax.scatter(np.zeros(results.shape[1])+nops, results[nops], s=1, c='blue')

                #print('{:4}/{}\t{:.9f}'.format(nops, results.shape[0], np.sum(results[nops]) / results.shape[1]))
                res.append(np.sum(results[nops]) / results.shape[1])
            #ax.set_ylim(auto=True)
            #ax.set_xlim(min_nops, max_nops)
            #fig.savefig(f'benchmarks/tmu_load_2_slot_1_qpu_{unif[2]}_{unif[3]}.png')
    return res

def getHwAddr(ifname):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    info = fcntl.ioctl(s.fileno(), 0x8927,  struct.pack('256s', bytes(ifname, 'utf-8')[:15]))
    return ':'.join('%02x' % b for b in info[18:24])


#for x in range(0,10):
def main():
	#for n in range(0,100):
		#
		s=int(sys.argv[1])
		r=int(sys.argv[2])
		
		#f=sys.argv[2]
		mac=getHwAddr('eth0')
		results=[]
		#results.append(c)
		#results.append(f)
		results.append(os.popen("vcgencmd measure_temp | cut -d = -f 2 | cut -d \"'\" -f 1").read()[:-1])
		
		results.append(get_QPU_freq(s))
		
		for i in test_clock():
                	results.append(i)
		for i in test_clock():
                	results.append(i)
		
		results.append(cpu_hash())
		#results.append(os.popen("vcgencmd measure_clock core").read[:-1])
		results.append(cpu_random())
		results.append(cpu_true_random(r))
		
		
		"""results.append(get_QPU_freq(1))
		results.append(get_QPU_freq(2))
		results.append(get_QPU_freq(5))
		results.append(get_QPU_freq(7))
		results.append(get_QPU_freq(8))
		results.append(get_QPU_freq(10))
		results.append(get_QPU_freq(60))"""

		"""results.append(cpu_true_random(1000))
		results.append(cpu_true_random(1000000))
		results.append(cpu_true_random(100000000))
		for i in test_clock():
			results.append(i)
		for i in test_clock():
			results.append(i)
		for i in sgemm_rnn_naive():
			results.append(i)
		for i in summation(length=32 * 1024 * 1024):
			results.append(i)
		for i in scopy(length=16*1024*1024):
			results.append(i)
		for i in test_multiple_dispatch_delay():
			results.append(i)"""
		#for i in test_tmu_load_1_slot_1_qpu():
		#	results.append(i)
		#for i in test_tmu_load_2_slot_1_qpu():
		#	results.append(i)
		
		results.append(mac)
		print(*results, sep=',')
		#print(memset(fill=0x5a5a5a5a, length=16 * 1024 * 1024))

if __name__ == "__main__":
    main()
