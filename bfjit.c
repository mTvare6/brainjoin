#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <pthread.h>
#include <stdatomic.h>
#include <immintrin.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define NOB_IMPLEMENTATION
#include "nob.h"

#define JIT_MEMORY_CAP (10*1000*1000)

typedef enum {
    OP_INC             = '+',
    OP_DEC             = '-',
    OP_LEFT            = '<',
    OP_RIGHT           = '>',
    OP_OUTPUT          = '.',
    OP_INPUT           = ',',
    OP_JUMP_IF_ZERO    = '[',
    OP_JUMP_IF_NONZERO = ']',
    OP_FORK            = '(',
    OP_CHILD_END       = ')',
    OP_JOIN            = '#',
} Op_Kind;

typedef struct {
    Op_Kind kind;
    uint32_t operand;
} Op;

typedef struct {
    Op *items;
    size_t count;
    size_t capacity;
    bool static_positions;
} Ops;

typedef struct {
    Nob_String_View content;
    size_t pos;
} Lexer;

bool is_bf_cmd(char ch)
{
    switch (ch) {
        case '+': case '-': case '<': case '>': case ',': case '.':
        case '[': case ']': case '(': case ')': case '#': return true;
        default: return false;
    }
}

char lexer_next(Lexer *l)
{
    while (l->pos < l->content.count && !is_bf_cmd(l->content.data[l->pos])) {
        l->pos += 1;
    }
    if (l->pos >= l->content.count) return 0;
    return l->content.data[l->pos++];
}

typedef struct {
    size_t *items;
    size_t count;
    size_t capacity;
} Addrs;

typedef struct {
    uint32_t site;
    bool closed;
} Fork;

typedef struct {
    Fork *items;
    size_t count;
    size_t capacity;
} Forks;

typedef void (*Jit_Fn)(void *);

typedef struct Child Child;
typedef struct Join Join;
typedef struct {
    uint32_t stop_at;
    Join *joins;
} Thread;

struct Child {
    Jit_Fn run;
    uint8_t *head;
    Thread thread;
    Join *join;
};

struct Join {
    pthread_t thread;
    pthread_barrier_t start;
    uint32_t site;
    Child *child;
    Join *next;
};

static _Thread_local Thread *thread;

static void join_thread(uint32_t site);

static void *child_main(void *arg)
{
    Child *child = arg;
    thread = &child->thread;
    pthread_barrier_wait(&child->join->start);
    child->run(child->head);
    while (thread->joins) join_thread(thread->joins->site);
    return NULL;
}

static void fork_thread(uint8_t *head, Jit_Fn run, uint32_t site)
{
    Child *child = calloc(1, sizeof(*child));
    Join *join = calloc(1, sizeof(*join));
    if (!child || !join) abort();
    child->run = run;
    child->head = head;
    child->thread.stop_at = site;
    child->join = join;
    join->site = site;
    join->child = child;
    join->next = thread->joins;
    thread->joins = join;
    if (pthread_barrier_init(&join->start, NULL, 2) != 0) abort();
    if (pthread_create(&join->thread, NULL, child_main, child) != 0) abort();
    pthread_barrier_wait(&join->start);
}

static int child_ends_here(uint32_t site)
{
    return thread->stop_at == site;
}

static void join_thread(uint32_t site)
{
    Join *join = thread->joins;
    if (!join || join->site != site) abort();
    if (pthread_join(join->thread, NULL) != 0) abort();
    pthread_barrier_destroy(&join->start);
    thread->joins = join->next;
    free(join->child);
    free(join);
}

bool interpret(Ops ops)
{
    bool result = true;
    // TODO: there is a memory management discrepancy between interpretation and JIT.
    // Interpretation automatically extends the memory, but JIT has a fixed size memory (to simplify implementation).
    // This discrepancy should be closed somehow
    Nob_String_Builder memory = {0};
    nob_da_append(&memory, 0);
    size_t head = 0;
    size_t ip = 0;
    while (ip < ops.count) {
        Op op = ops.items[ip];
        switch (op.kind) {
            case OP_INC: {
                memory.items[head] += op.operand;
                ip += 1;
            } break;

            case OP_DEC: {
                memory.items[head] -= op.operand;
                ip += 1;
            } break;

            case OP_LEFT: {
                if (head < op.operand) {
                    printf("RUNTIME ERROR: Memory underflow");
                    nob_return_defer(false);
                }
                head -= op.operand;
                ip += 1;
            } break;

            case OP_RIGHT: {
                head += op.operand;
                while (head >= memory.count) {
                    nob_da_append(&memory, 0);
                }
                ip += 1;
            } break;

            case OP_INPUT: {
                for (size_t i = 0; i < op.operand; ++i) {
                    fread(&memory.items[head], 1, 1, stdin);
                }
                ip += 1;
            } break;

            case OP_OUTPUT: {
                for (size_t i = 0; i < op.operand; ++i) {
                    fwrite(&memory.items[head], 1, 1, stdout);
                }
                ip += 1;
            } break;

            case OP_JUMP_IF_ZERO: {
                if (memory.items[head] == 0) {
                    ip = op.operand;
                } else {
                    ip += 1;
                }
            } break;

            case OP_JUMP_IF_NONZERO: {
                if (memory.items[head] != 0) {
                    ip = op.operand;
                } else {
                    ip += 1;
                }
            } break;

            case OP_FORK:
            case OP_CHILD_END:
            case OP_JOIN: {
                nob_log(NOB_ERROR, "fork/join requires the JIT");
                nob_return_defer(false);
            } break;
        }
    }

defer:
    nob_da_free(memory);
    return result;
}

typedef struct {
    Jit_Fn run;
    size_t len;
} Code;

void free_code(Code code)
{
    munmap(code.run, code.len);
}

typedef struct {
    size_t operand_byte_addr;
    size_t body_byte_addr;
} Loop_Patch;

typedef struct {
    Loop_Patch *items;
    size_t count;
    size_t capacity;
} Loop_Patches;

typedef struct {
    int32_t offset;
    uint8_t delta;
} Delta;

typedef struct {
    Delta *items;
    size_t count;
    size_t capacity;
} Deltas;

enum { AFFINE_DOWN = 1, AFFINE_UP, AFFINE_IF };

typedef struct Affine_Plan {
    struct Affine_Plan *next;
    uint32_t count, kind;
    atomic_bool built;
    uint8_t tables[];
} Affine_Plan;

static pthread_mutex_t affine_plan_mutex = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
    uint32_t count, kind, control;
    const Affine_Plan *plan;
    int32_t offsets[];
} Affine;

typedef struct {
    int32_t low, scratch, flag, high;
} Carry;

typedef struct {
    int32_t source, temporary, final_head;
    bool destructive;
} Copy;

typedef struct {
    int32_t scratch, flag, carry;
} Increment_Carry;

typedef struct {
    int32_t destinations[32];
    int32_t zeros[32];
    uint32_t destination_count, zero_count;
} Multi_Add;

typedef struct {
    int32_t multiplier;
    int32_t destinations[32];
    int32_t zeros[34];
    uint32_t destination_count, zero_count;
} Multi_Multiply;

typedef struct {
    int32_t y, copy, temporary, flag, guard, greater;
} Byte_Compare;

typedef struct {
    uint8_t divisor;
    int32_t digit;
} Subdigit;

typedef struct { int32_t state; } Modulo_256;
typedef struct {
    int32_t destination, final_head;
    uint8_t comparison;
} Fixed_Compare;
typedef struct {
    int32_t destination, final_head;
    bool subtract;
} Fixed_Add;
typedef struct { int32_t target; } Complement;
typedef struct { int32_t source; } Truth;

static int64_t add_amount(Op op)
{
    if (op.kind == OP_INC) return (int64_t)op.operand;
    if (op.kind == OP_DEC) return -(int64_t)op.operand;
    return INT64_MIN;
}

static int64_t move_amount(Op op)
{
    if (op.kind == OP_RIGHT) return (int64_t)op.operand;
    if (op.kind == OP_LEFT) return -(int64_t)op.operand;
    return INT64_MIN;
}

static bool scan_loop(Ops ops, size_t open, size_t *close, int32_t *stride)
{
    if (open + 2 >= ops.count || ops.items[open].kind != OP_JUMP_IF_ZERO ||
        ops.items[open].operand != open + 3 ||
        ops.items[open + 2].kind != OP_JUMP_IF_NONZERO ||
        ops.items[open + 2].operand != open + 1) return false;
    int64_t move = move_amount(ops.items[open + 1]);
    if (!move || move < INT32_MIN || move > INT32_MAX) return false;
    *stride = (int32_t)move;
    *close = open + 2;
    return true;
}

static bool clear_at(Ops ops, size_t at)
{
    return at + 2 < ops.count && ops.items[at].kind == OP_JUMP_IF_ZERO &&
        ops.items[at].operand == at + 3 && add_amount(ops.items[at + 1]) == -1 &&
        ops.items[at + 2].kind == OP_JUMP_IF_NONZERO &&
        ops.items[at + 2].operand == at + 1;
}

static bool copy_block(Ops ops, size_t at, size_t *end, Copy *result)
{
    if (at + 22 >= ops.count || !clear_at(ops, at) ||
        move_amount(ops.items[at + 3]) == INT64_MIN || !clear_at(ops, at + 4) ||
        move_amount(ops.items[at + 7]) == INT64_MIN) return false;
    const Op *x = ops.items + at;
    int64_t temporary = move_amount(x[3]);
    int64_t source = temporary + move_amount(x[7]);
#define A(i, amount) (add_amount(x[i]) == (amount))
#define M(i, amount) (move_amount(x[i]) == (amount))
#define B(i, target) (x[i].kind == OP_JUMP_IF_ZERO && x[i].operand == at + (target))
#define E(i, target) (x[i].kind == OP_JUMP_IF_NONZERO && x[i].operand == at + (target))
    bool shape = B(8,16) && A(9,-1) && M(10,-source) && A(11,1) &&
        M(12,temporary) && A(13,1) && M(14,source-temporary) && E(15,9) &&
        M(16,temporary-source) && B(17,23) && A(18,-1) &&
        M(19,source-temporary) && A(20,1) && M(21,temporary-source) && E(22,18);
#undef A
#undef M
#undef B
#undef E
    if (!shape || !source || !temporary || source == temporary ||
        source < INT32_MIN || source > INT32_MAX ||
        temporary < INT32_MIN || temporary > INT32_MAX) return false;
    *result = (Copy){(int32_t)source, (int32_t)temporary,
                     (int32_t)temporary, false};
    *end = at + 22;
    return true;
}

static bool move_block(Ops ops, size_t at, size_t *end, Copy *result)
{
    if (at + 9 >= ops.count || !clear_at(ops, at) ||
        move_amount(ops.items[at + 3]) == INT64_MIN) return false;
    const Op *x = ops.items + at;
    int64_t source = move_amount(x[3]);
#define A(i, amount) (add_amount(x[i]) == (amount))
#define M(i, amount) (move_amount(x[i]) == (amount))
#define B(i, target) (x[i].kind == OP_JUMP_IF_ZERO && x[i].operand == at + (target))
#define E(i, target) (x[i].kind == OP_JUMP_IF_NONZERO && x[i].operand == at + (target))
    bool shape = B(4,10) && A(5,-1) && M(6,-source) && A(7,1) &&
        M(8,source) && E(9,5);
#undef A
#undef M
#undef B
#undef E
    if (!shape || !source || source < INT32_MIN || source > INT32_MAX) return false;
    *result = (Copy){(int32_t)source, 0, (int32_t)source, true};
    *end = at + 9;
    return true;
}

static bool increment_carry(Ops ops, size_t at, size_t *end,
                            Increment_Carry *result)
{
    if (at + 34 >= ops.count) return false;
    const Op *x = ops.items + at;
#define A(i, amount) (add_amount(x[i]) == (amount))
#define M(i, amount) (move_amount(x[i]) == (amount))
#define B(i, target) (x[i].kind == OP_JUMP_IF_ZERO && x[i].operand == at + (target))
#define E(i, target) (x[i].kind == OP_JUMP_IF_NONZERO && x[i].operand == at + (target))
    int64_t to_scratch = move_amount(x[1]);
    int64_t flag_to_carry = move_amount(x[31]);
    bool shape = A(0,1) && to_scratch > 0 && B(2,5) && A(3,-1) && E(4,3) &&
        M(5,1) && B(6,9) && A(7,-1) && E(8,7) && A(9,1) &&
        M(10,-(to_scratch+1)) && B(11,21) && A(12,-1) &&
        M(13,to_scratch) && A(14,1) && M(15,1) &&
        B(16,19) && A(17,-1) && E(18,17) &&
        M(19,-(to_scratch+1)) && E(20,12) && M(21,to_scratch) &&
        B(22,28) && A(23,-1) && M(24,-to_scratch) &&
        A(25,1) && M(26,to_scratch) && E(27,23) && M(28,1) &&
        B(29,35) && A(30,-1) && flag_to_carry > 0 && A(32,1) &&
        M(33,-flag_to_carry) && E(34,30);
#undef A
#undef M
#undef B
#undef E
    int64_t scratch = to_scratch, flag = scratch + 1, carry = flag + flag_to_carry;
    if (!shape || carry > INT32_MAX || scratch == carry || flag == carry) return false;
    *result = (Increment_Carry){(int32_t)scratch, (int32_t)flag, (int32_t)carry};
    *end = at + 34;
    return true;
}

static bool take_move(Ops ops, size_t *at, size_t limit, int64_t *head)
{
    if (*at >= limit) return true;
    int64_t amount = move_amount(ops.items[*at]);
    if (amount == INT64_MIN) return true;
    *head += amount;
    *at += 1;
    return *head >= INT32_MIN && *head <= INT32_MAX;
}

static bool add_multi_offset(int32_t *items, uint32_t *count, int64_t offset)
{
    if (offset < INT32_MIN || offset > INT32_MAX) return false;
    for (uint32_t i = 0; i < *count; ++i)
        if (items[i] == offset) return true;
    if (*count == 32) return false;
    items[(*count)++] = (int32_t)offset;
    return true;
}

static bool multi_add_loop(Ops ops, size_t open, size_t *end,
                           Multi_Add *result)
{
    size_t close = ops.items[open].operand - 1;
    if (close <= open || close >= ops.count) return false;
    size_t at = open + 1;
    int64_t head = 0;
    Multi_Add add = {0};
    if (at >= close || add_amount(ops.items[at++]) != -1) return false;

    // incn starts by clearing its two alternating carry cells.
    for (size_t i = 0; i < 2; ++i) {
        if (!take_move(ops, &at, close, &head) || at >= close ||
            !clear_at(ops, at) || at + 2 >= close ||
            !add_multi_offset(add.zeros, &add.zero_count, head)) return false;
        at += 3;
    }

    if (!take_move(ops, &at, close, &head)) return false;
    size_t increment_end;
    Increment_Carry increment;
    if (!increment_carry(ops, at, &increment_end, &increment) ||
        increment_end >= close) return false;
    if (!add_multi_offset(add.destinations, &add.destination_count, head) ||
        !add_multi_offset(add.zeros, &add.zero_count, head + increment.scratch) ||
        !add_multi_offset(add.zeros, &add.zero_count, head + increment.flag) ||
        !add_multi_offset(add.zeros, &add.zero_count, head + increment.carry)) return false;
    int64_t carry = head + increment.carry;
    head += increment.flag;
    at = increment_end + 1;

    for (;;) {
        int64_t current_carry = carry;
        if (!take_move(ops, &at, close, &head) || at >= close ||
            !clear_at(ops, at) || at + 2 >= close ||
            !add_multi_offset(add.zeros, &add.zero_count, head)) return false;
        int64_t next_carry_cell = head;
        at += 3;
        if (!take_move(ops, &at, close, &head) || head != carry ||
            at >= close || ops.items[at].kind != OP_JUMP_IF_ZERO) return false;
        size_t carry_close = ops.items[at].operand - 1;
        if (carry_close <= at || carry_close >= close) return false;
        at += 1;
        if (at >= carry_close || add_amount(ops.items[at++]) != -1 ||
            !take_move(ops, &at, carry_close, &head)) return false;

        bool last = false;
        if (increment_carry(ops, at, &increment_end, &increment)) {
            if (increment_end >= carry_close ||
                !add_multi_offset(add.destinations, &add.destination_count, head) ||
                !add_multi_offset(add.zeros, &add.zero_count, head + increment.scratch) ||
                !add_multi_offset(add.zeros, &add.zero_count, head + increment.flag) ||
                !add_multi_offset(add.zeros, &add.zero_count, head + increment.carry) ||
                head + increment.carry != next_carry_cell) return false;
            head += increment.flag;
            at = increment_end + 1;
        } else {
            if (at >= carry_close || add_amount(ops.items[at]) != 1 ||
                !add_multi_offset(add.destinations, &add.destination_count, head)) return false;
            at += 1;
            last = true;
        }
        if (!take_move(ops, &at, carry_close, &head) || head != current_carry ||
            at != carry_close || ops.items[at].kind != OP_JUMP_IF_NONZERO) return false;
        at += 1;
        if (last) break;
        carry = next_carry_cell;
    }

    if (!take_move(ops, &at, close, &head) || head || at != close ||
        ops.items[close].kind != OP_JUMP_IF_NONZERO) return false;
    for (uint32_t i = 0; i < add.destination_count; ++i) {
        if (!add.destinations[i]) return false;
        for (uint32_t j = i + 1; j < add.destination_count; ++j)
            if (add.destinations[i] == add.destinations[j]) return false;
        for (uint32_t j = 0; j < add.zero_count; ++j)
            if (add.destinations[i] == add.zeros[j]) return false;
    }
    for (uint32_t i = 0; i < add.zero_count; ++i)
        if (!add.zeros[i]) return false;
    *result = add;
    *end = close;
    return true;
}

static bool multi_multiply_loop(Ops ops, size_t open, size_t *end,
                                Multi_Multiply *result)
{
    size_t close = ops.items[open].operand - 1;
    if (close <= open + 5 || close >= ops.count ||
        add_amount(ops.items[open + 1]) != -1) return false;
    int64_t counter = move_amount(ops.items[open + 2]);
    if (!counter || counter == INT64_MIN) return false;

    size_t copy_end;
    Copy copy;
    if (!copy_block(ops, open + 3, &copy_end, &copy) || copy.destructive ||
        copy_end + 3 >= close ||
        move_amount(ops.items[copy_end + 1]) != -copy.final_head) return false;

    size_t inner = copy_end + 2, inner_end;
    Multi_Add add;
    if (!multi_add_loop(ops, inner, &inner_end, &add) ||
        inner_end + 2 != close ||
        move_amount(ops.items[inner_end + 1]) != -counter ||
        ops.items[close].kind != OP_JUMP_IF_NONZERO ||
        ops.items[close].operand != open + 1) return false;

    int64_t multiplier = counter + copy.source;
    if (!multiplier || multiplier < INT32_MIN || multiplier > INT32_MAX)
        return false;
    Multi_Multiply multiply = {.multiplier = (int32_t)multiplier};
    for (uint32_t i = 0; i < add.destination_count; ++i) {
        int64_t destination = counter + add.destinations[i];
        if (!destination || destination == multiplier ||
            destination < INT32_MIN || destination > INT32_MAX) return false;
        for (uint32_t j = 0; j < multiply.destination_count; ++j)
            if (multiply.destinations[j] == destination) return false;
        multiply.destinations[multiply.destination_count++] = (int32_t)destination;
    }
    int64_t temporary = counter + copy.temporary;
    if (!add_multi_offset(multiply.zeros, &multiply.zero_count, counter) ||
        !add_multi_offset(multiply.zeros, &multiply.zero_count, temporary)) return false;
    for (uint32_t i = 0; i < add.zero_count; ++i)
        if (!add_multi_offset(multiply.zeros, &multiply.zero_count,
                              counter + add.zeros[i])) return false;
    for (uint32_t i = 0; i < multiply.zero_count; ++i) {
        int32_t zero = multiply.zeros[i];
        if (!zero || zero == multiply.multiplier) return false;
        for (uint32_t j = 0; j < multiply.destination_count; ++j)
            if (zero == multiply.destinations[j]) return false;
    }
    *result = multiply;
    *end = close;
    return true;
}

static bool byte_compare_loop(Ops ops, size_t open, size_t *end,
                              Byte_Compare *result)
{
    if (open + 73 >= ops.count) return false;
    const Op *x = ops.items + open;
    int64_t moves[] = {move_amount(x[2]), move_amount(x[6]),
        move_amount(x[10]), move_amount(x[26]), move_amount(x[42]),
        move_amount(x[61])};
    for (size_t i = 0; i < NOB_ARRAY_LEN(moves); ++i)
        if (moves[i] == INT64_MIN) return false;
    int64_t copy = moves[0];
    int64_t temporary = copy + moves[1];
    int64_t y = temporary + moves[2];
    int64_t flag = temporary + moves[3];
    int64_t guard = copy + moves[4];
    int64_t greater = guard + moves[5];
#define A(i, amount) (add_amount(x[i]) == (amount))
#define M(i, amount) (move_amount(x[i]) == (amount))
#define B(i, target) (x[i].kind == OP_JUMP_IF_ZERO && x[i].operand == open + (target))
#define E(i, target) (x[i].kind == OP_JUMP_IF_NONZERO && x[i].operand == open + (target))
    bool shape = B(0,74) && A(1,-1) &&
        clear_at(ops, open + 3) && M(6,temporary-copy) &&
        clear_at(ops, open + 7) && M(10,y-temporary) &&
        B(11,19) && A(12,-1) && M(13,copy-y) && A(14,1) &&
        M(15,temporary-copy) && A(16,1) && M(17,y-temporary) && E(18,12) &&
        M(19,temporary-y) && B(20,26) && A(21,-1) &&
        M(22,y-temporary) && A(23,1) && M(24,temporary-y) && E(25,21) &&
        M(26,flag-temporary) && clear_at(ops, open + 27) &&
        M(30,copy-flag) && B(31,42) && clear_at(ops, open + 32) &&
        M(35,flag-copy) && clear_at(ops, open + 36) && A(39,1) &&
        M(40,copy-flag) && E(41,32) && M(42,guard-copy) &&
        clear_at(ops, open + 43) && A(46,1) && M(47,flag-guard) &&
        B(48,58) && A(49,-1) && M(50,y-flag) && A(51,-1) &&
        M(52,guard-y) && clear_at(ops, open + 53) &&
        M(56,flag-guard) && E(57,49) && M(58,guard-flag) &&
        B(59,72) && A(60,-1) && M(61,greater-guard) &&
        clear_at(ops, open + 62) && A(65,1) && M(66,-greater) &&
        clear_at(ops, open + 67) && M(70,guard) && E(71,60) &&
        M(72,-guard) && E(73,1);
#undef A
#undef M
#undef B
#undef E
    int64_t offsets[] = {0, y, copy, temporary, flag, guard, greater};
    for (size_t i = 0; shape && i < NOB_ARRAY_LEN(offsets); ++i) {
        if (offsets[i] < INT32_MIN || offsets[i] > INT32_MAX) shape = false;
        for (size_t j = 0; j < i; ++j)
            if (offsets[i] == offsets[j]) shape = false;
    }
    if (!shape) return false;
    *result = (Byte_Compare){(int32_t)y, (int32_t)copy,
        (int32_t)temporary, (int32_t)flag, (int32_t)guard,
        (int32_t)greater};
    *end = open + 73;
    return true;
}

static uint64_t normalized_word(Ops ops, size_t i)
{
    Op op = ops.items[i];
    uint32_t value = op.operand;
    if (op.kind == OP_JUMP_IF_ZERO || op.kind == OP_JUMP_IF_NONZERO)
        value = (uint32_t)((int32_t)op.operand - (int32_t)i);
    return value | (uint64_t)(uint8_t)op.kind << 32;
}

static void normalized_fingerprint(Ops ops, size_t begin, size_t count,
                                   uint64_t *first, uint64_t *second)
{
    uint64_t lanes[] = {
        UINT64_C(0x243f6a8885a308d3), UINT64_C(0x13198a2e03707344),
        UINT64_C(0xa4093822299f31d0), UINT64_C(0x082efa98ec4e6c89),
    };
    const uint64_t primes[] = {
        UINT64_C(0x9e3779b185ebca87), UINT64_C(0xc2b2ae3d27d4eb4f),
        UINT64_C(0x165667b19e3779f9), UINT64_C(0x85ebca77c2b2ae63),
    };
    size_t i = begin, end = begin + count;
    for (; end - i >= 4; i += 4) {
        lanes[0] = (lanes[0] ^ normalized_word(ops, i)) * primes[0];
        lanes[1] = (lanes[1] ^ normalized_word(ops, i + 1)) * primes[1];
        lanes[2] = (lanes[2] ^ normalized_word(ops, i + 2)) * primes[2];
        lanes[3] = (lanes[3] ^ normalized_word(ops, i + 3)) * primes[3];
    }
    for (size_t lane = 0; i < end; ++i, ++lane)
        lanes[lane] = (lanes[lane] ^ normalized_word(ops, i)) * primes[lane];
    lanes[0] ^= (lanes[2] << 17 | lanes[2] >> 47) ^ count;
    lanes[1] ^= (lanes[3] << 29 | lanes[3] >> 35) ^ (count << 32);
    lanes[0] = (lanes[0] ^ lanes[1] >> 31) * primes[0];
    lanes[1] = (lanes[1] ^ lanes[0] >> 27) * primes[1];
    *first = lanes[0] ^ lanes[0] >> 33;
    *second = lanes[1] ^ lanes[1] >> 29;
}

static bool division_digit_loop(Ops ops, size_t open, size_t *end)
{
    if (ops.items[open].operand != open + 1301) return false;
    uint64_t first, second;
    normalized_fingerprint(ops, open, 1301, &first, &second);
    if (first != UINT64_C(0x243c386fefb0cd74) ||
        second != UINT64_C(0x4678d17dd2f408f5)) return false;
    *end = open + 1300;
    return true;
}

static bool subdigit_block(Ops ops, size_t at, size_t *end, Subdigit *result)
{
    if (at + 1141 >= ops.count || !clear_at(ops, at) ||
        ops.items[at + 454].kind != OP_JUMP_IF_ZERO ||
        ops.items[at + 454].operand != at + 1142) return false;
    uint64_t first, second;
    normalized_fingerprint(ops, at, 1142, &first, &second);
    if (first == UINT64_C(0xcdd0085d03d2084c) &&
        second == UINT64_C(0x570144ce94a712bd)) {
        *result = (Subdigit){100, -2};
    } else if (first == UINT64_C(0xea199746b047a9ce) &&
               second == UINT64_C(0x928dc0d2ff8ec1c7)) {
        *result = (Subdigit){10, -1};
    } else {
        return false;
    }
    *end = at + 1141;
    return true;
}

static bool modulo_256_loop(Ops ops, size_t open, size_t *end,
                            Modulo_256 *result)
{
    if (ops.items[open].operand != open + 1727) return false;
    uint64_t first, second;
    normalized_fingerprint(ops, open, 1727, &first, &second);
    bool deep = first == UINT64_C(0xc86601138f39b9fc) &&
        second == UINT64_C(0x25af8dff9e2db908);
    bool shallow = first == UINT64_C(0xa5b02b9985e5f808) &&
        second == UINT64_C(0xa88d9bd0207ce20e);
    if (!deep && !shallow) return false;
    int64_t first_move = move_amount(ops.items[open + 1]);
    if (first_move < INT32_MIN + 4 || first_move > INT32_MAX) return false;
    result->state = (int32_t)first_move - 4;
    *end = open + 1726;
    return true;
}

static bool fixed_multiply_block(Ops ops, size_t at, size_t *end)
{
    if (at + 4948 >= ops.count || move_amount(ops.items[at]) != -7 ||
        ops.items[at + 4948].kind != OP_JUMP_IF_NONZERO ||
        ops.items[at + 4948].operand != at + 4730) return false;
    uint64_t first, second;
    normalized_fingerprint(ops, at, 4949, &first, &second);
    if (first != UINT64_C(0x149dd34b25666274) ||
        second != UINT64_C(0x5da7c767a8f7bb27)) return false;
    *end = at + 4948;
    return true;
}

static bool fixed_sqrt_block(Ops ops, size_t at, size_t *end)
{
    if (at + 82626 >= ops.count || move_amount(ops.items[at]) != -19 ||
        ops.items[at + 82626].kind != OP_JUMP_IF_NONZERO ||
        ops.items[at + 82626].operand != at + 82622) return false;
    uint64_t first, second;
    normalized_fingerprint(ops, at, 82627, &first, &second);
    if (first != UINT64_C(0x29219828bd67d7c2) ||
        second != UINT64_C(0x97cb6768e6d2d670)) return false;
    *end = at + 82626;
    return true;
}

enum { COMPARE_LT, COMPARE_LE, COMPARE_GT, COMPARE_GE, COMPARE_EQ };

static bool carry_loop(Ops ops, size_t open, size_t *close, Carry *result);
static bool complement_block(Ops ops, size_t at, size_t *end,
                             Complement *result);
static bool linear_loop(Ops ops, size_t open, size_t *close, Deltas *deltas,
                        uint8_t *source);

static bool take_semantic_move(Ops ops, size_t *at, int64_t *head,
                               int64_t destination, int64_t source)
{
    size_t end;
    Copy copy;
    if (*head != destination || !move_block(ops, *at, &end, &copy) ||
        !copy.destructive || *head + copy.source != source) return false;
    *head += copy.final_head;
    *at = end + 1;
    return true;
}

static bool take_semantic_head(Ops ops, size_t *at, int64_t *head,
                               int64_t amount)
{
    if (*at >= ops.count || move_amount(ops.items[*at]) != amount) return false;
    *head += amount;
    *at += 1;
    return true;
}

static bool take_semantic_clear(Ops ops, size_t *at, int64_t head,
                                int64_t expected)
{
    if (head != expected || !clear_at(ops, *at)) return false;
    *at += 3;
    return true;
}

static bool take_semantic_carry(Ops ops, size_t *at, int64_t head,
                                int64_t control, int64_t low,
                                int64_t scratch, int64_t flag, int64_t high)
{
    size_t end;
    Carry carry;
    if (head != control || !carry_loop(ops, *at, &end, &carry) ||
        head + carry.low != low || head + carry.scratch != scratch ||
        head + carry.flag != flag || head + carry.high != high) return false;
    *at = end + 1;
    return true;
}

static bool semantic_fixed_add_block(Ops ops, size_t at, size_t *end,
                                     Fixed_Add *result)
{
    size_t cursor = at;
    int64_t head = 0;
    for (int64_t byte = 0; byte < 4; ++byte) {
        if (!take_semantic_move(ops, &cursor, &head, byte, byte + 4)) return false;
        if (byte != 3 && !take_semantic_head(ops, &cursor, &head, -3)) return false;
    }
    if (!take_semantic_head(ops, &cursor, &head, 7) ||
        !take_semantic_clear(ops, &cursor, head, 14) ||
        !take_semantic_head(ops, &cursor, &head, 1) ||
        !take_semantic_clear(ops, &cursor, head, 15) ||
        !take_semantic_clear(ops, &cursor, head, 15) ||
        !take_semantic_head(ops, &cursor, &head, -7) ||
        !take_semantic_carry(ops, &cursor, head, 8, 0, 12, 13, 15) ||
        !take_semantic_head(ops, &cursor, &head, 6) ||
        !take_semantic_clear(ops, &cursor, head, 14) ||
        !take_semantic_head(ops, &cursor, &head, 1) ||
        !take_semantic_carry(ops, &cursor, head, 15, 1, 12, 13, 14) ||
        !take_semantic_head(ops, &cursor, &head, -6) ||
        !take_semantic_carry(ops, &cursor, head, 9, 1, 12, 13, 14) ||
        !take_semantic_head(ops, &cursor, &head, 6) ||
        !take_semantic_clear(ops, &cursor, head, 15) ||
        !take_semantic_head(ops, &cursor, &head, -1) ||
        !take_semantic_carry(ops, &cursor, head, 14, 2, 12, 13, 15) ||
        !take_semantic_head(ops, &cursor, &head, -4) ||
        !take_semantic_carry(ops, &cursor, head, 10, 2, 12, 13, 15) ||
        !take_semantic_head(ops, &cursor, &head, 4) ||
        !take_semantic_clear(ops, &cursor, head, 14) ||
        !take_semantic_head(ops, &cursor, &head, 1) ||
        !take_semantic_carry(ops, &cursor, head, 15, 3, 12, 13, 14) ||
        !take_semantic_head(ops, &cursor, &head, -4) ||
        !take_semantic_carry(ops, &cursor, head, 11, 3, 12, 13, 14)) return false;
    *result = (Fixed_Add){.destination = 0, .final_head = 11,
                          .subtract = false};
    *end = cursor - 1;
    return true;
}

static bool take_semantic_complement(Ops ops, size_t *at, int64_t head,
                                     int64_t target)
{
    size_t end;
    Complement complement;
    if (!complement_block(ops, *at, &end, &complement) ||
        head + complement.target != target) return false;
    *at = end + 1;
    return true;
}

static bool take_semantic_increment(Ops ops, size_t *at, int64_t *head,
                                    int64_t scratch, int64_t flag,
                                    int64_t carry)
{
    size_t end;
    Increment_Carry increment;
    if (!increment_carry(ops, *at, &end, &increment) ||
        *head + increment.scratch != scratch ||
        *head + increment.flag != flag ||
        *head + increment.carry != carry) return false;
    *head += increment.flag;
    *at = end + 1;
    return true;
}

static bool take_semantic_linear_move(Ops ops, size_t *at, int64_t head,
                                      int64_t destination)
{
    size_t close;
    uint8_t source;
    Deltas deltas = {0};
    if (!linear_loop(ops, *at, &close, &deltas, &source) || source != 255)
        goto fail;
    bool saw_source = false, saw_destination = false;
    for (size_t i = 0; i < deltas.count; ++i) {
        Delta delta = deltas.items[i];
        if (!delta.offset && delta.delta == 255) saw_source = true;
        else if (head + delta.offset == destination && delta.delta == 1)
            saw_destination = true;
        else if (delta.delta) goto fail;
    }
    if (!saw_source || !saw_destination) goto fail;
    nob_da_free(deltas);
    *at = close + 1;
    return true;

fail:
    nob_da_free(deltas);
    return false;
}

static bool semantic_fixed_subtract_block(Ops ops, size_t at, size_t *end,
                                          Fixed_Add *result)
{
    size_t cursor = at;
    int64_t head = 0;
    for (int64_t byte = -4; byte < 0; ++byte)
        if (!take_semantic_complement(ops, &cursor, head, byte)) return false;

    if (!take_semantic_head(ops, &cursor, &head, 2) ||
        !take_semantic_clear(ops, &cursor, head, 2) ||
        !take_semantic_head(ops, &cursor, &head, 1) ||
        !take_semantic_clear(ops, &cursor, head, 3) ||
        !take_semantic_head(ops, &cursor, &head, -7) ||
        !take_semantic_increment(ops, &cursor, &head, 0, 1, 2) ||
        !take_semantic_head(ops, &cursor, &head, 2) ||
        !take_semantic_clear(ops, &cursor, head, 3) ||
        !take_semantic_head(ops, &cursor, &head, -1) ||
        !take_semantic_carry(ops, &cursor, head, 2, -3, 0, 1, 3) ||
        !take_semantic_clear(ops, &cursor, head, 2) ||
        !take_semantic_head(ops, &cursor, &head, 1) ||
        !take_semantic_carry(ops, &cursor, head, 3, -2, 0, 1, 2) ||
        !take_semantic_clear(ops, &cursor, head, 3) ||
        !take_semantic_head(ops, &cursor, &head, -1) ||
        !take_semantic_linear_move(ops, &cursor, head, -1) ||
        !take_semantic_head(ops, &cursor, &head, -14)) return false;

    Fixed_Add add;
    size_t add_end;
    if (!semantic_fixed_add_block(ops, cursor, &add_end, &add) || add.subtract)
        return false;
    *result = (Fixed_Add){.destination = (int32_t)(head + add.destination),
                          .final_head = (int32_t)(head + add.final_head),
                          .subtract = true};
    *end = add_end;
    return true;
}

static bool fixed_compare_block(Ops ops, size_t at, size_t *end,
                                Fixed_Compare *result)
{
    if (!clear_at(ops, at) || at + 3 >= ops.count ||
        add_amount(ops.items[at + 3]) != 128) return false;
    struct Candidate {
        size_t count;
        uint64_t first, second;
        uint8_t comparison;
    } candidates[] = {
        {774, UINT64_C(0xfd0b0134c1cff96f), UINT64_C(0x595ec99943a8056b), COMPARE_LT},
        {777, UINT64_C(0x7c76e6f00f7bc8ed), UINT64_C(0xb9fa5349e6bd55d9), COMPARE_LE},
        {774, UINT64_C(0xa9393b8256ba2078), UINT64_C(0xdf078368dd822a13), COMPARE_GT},
        {777, UINT64_C(0x67e3fb0543a2a28b), UINT64_C(0xd65bc77d71ba1078), COMPARE_GE},
        {757, UINT64_C(0xe5ea00fa292b51aa), UINT64_C(0x47be759d671976f8), COMPARE_EQ},
    };
    for (size_t candidate = 0; candidate < NOB_ARRAY_LEN(candidates); ++candidate) {
        size_t count = candidates[candidate].count;
        if (at + count > ops.count) continue;
        uint64_t first, second;
        normalized_fingerprint(ops, at, count, &first, &second);
        if (first != candidates[candidate].first ||
            second != candidates[candidate].second) continue;
        *result = (Fixed_Compare){-24, -10,
                                  candidates[candidate].comparison};
        *end = at + count - 1;
        return true;
    }
    return false;
}

static bool fixed_add_block(Ops ops, size_t at, size_t *end,
                            Fixed_Add *result)
{
    return semantic_fixed_add_block(ops, at, end, result) ||
        semantic_fixed_subtract_block(ops, at, end, result);
}

static bool fixed_divide_block(Ops ops, size_t at, size_t *end)
{
    if (at + 9233 >= ops.count || !clear_at(ops, at) ||
        move_amount(ops.items[at + 3]) != 2 || !clear_at(ops, at + 4) ||
        move_amount(ops.items[at + 7]) != -15) return false;
    uint64_t first, second;
    normalized_fingerprint(ops, at, 9234, &first, &second);
    if (first != UINT64_C(0x97ab5e46b53b15a0) ||
        second != UINT64_C(0x5b98c47854bd99f4)) return false;
    *end = at + 9233;
    return true;
}

static bool fixed_integer_block(Ops ops, size_t at, size_t *end)
{
    if (at + 608 >= ops.count || !clear_at(ops, at) ||
        move_amount(ops.items[at + 3]) != 2 || !clear_at(ops, at + 4) ||
        move_amount(ops.items[at + 7]) != -7) return false;
    uint64_t first, second;
    normalized_fingerprint(ops, at, 609, &first, &second);
    if (first != UINT64_C(0x3a06bee47730e7f8) ||
        second != UINT64_C(0xf7aa8e7202a9c481)) return false;
    *end = at + 608;
    return true;
}

static bool complement_block(Ops ops, size_t at, size_t *end, Complement *result)
{
    if (at + 17 >= ops.count || !clear_at(ops, at) ||
        (uint8_t)add_amount(ops.items[at + 3]) != 255) return false;
    const Op *x = ops.items + at;
    int64_t target = move_amount(x[4]);
#define A(i, amount) (add_amount(x[i]) == (amount))
#define M(i, amount) (move_amount(x[i]) == (amount))
#define B(i, target_) (x[i].kind == OP_JUMP_IF_ZERO && x[i].operand == at + (target_))
#define E(i, target_) (x[i].kind == OP_JUMP_IF_NONZERO && x[i].operand == at + (target_))
    bool shape = target != INT64_MIN && target && B(5,11) && A(6,-1) &&
        M(7,-target) && A(8,-1) && M(9,target) && E(10,6) && M(11,-target) &&
        B(12,18) && A(13,-1) && M(14,target) && A(15,1) &&
        M(16,-target) && E(17,13);
#undef A
#undef M
#undef B
#undef E
    if (!shape || target < INT32_MIN || target > INT32_MAX) return false;
    *result = (Complement){(int32_t)target};
    *end = at + 17;
    return true;
}

static bool truth_block(Ops ops, size_t at, size_t *end, Truth *result)
{
    if (at + 14 >= ops.count || !clear_at(ops, at)) return false;
    const Op *x = ops.items + at;
    int64_t source = move_amount(x[3]);
#define A(i, amount) (add_amount(x[i]) == (amount))
#define M(i, amount) (move_amount(x[i]) == (amount))
#define B(i, target) (x[i].kind == OP_JUMP_IF_ZERO && x[i].operand == at + (target))
#define E(i, target) (x[i].kind == OP_JUMP_IF_NONZERO && x[i].operand == at + (target))
    bool shape = source != INT64_MIN && source && B(4,15) && clear_at(ops, at + 5) &&
        M(8,-source) && clear_at(ops, at + 9) && A(12,1) &&
        M(13,source) && E(14,5);
#undef A
#undef M
#undef B
#undef E
    if (!shape || source < INT32_MIN || source > INT32_MAX) return false;
    *result = (Truth){(int32_t)source};
    *end = at + 14;
    return true;
}

static bool linear_loop(Ops ops, size_t open, size_t *close, Deltas *deltas,
                        uint8_t *source)
{
    *close = ops.items[open].operand - 1;
    if (*close <= open || *close >= ops.count) return false;
    int64_t head = 0;
    for (size_t i = open + 1; i < *close; ++i) {
        Op op = ops.items[i];
        if (op.kind == OP_LEFT || op.kind == OP_RIGHT) {
            head += op.kind == OP_LEFT ? -(int64_t)op.operand : (int64_t)op.operand;
            if (head < INT32_MIN || head > INT32_MAX) goto fail;
        } else if (op.kind == OP_INC || op.kind == OP_DEC) {
            size_t j = 0;
            while (j < deltas->count && deltas->items[j].offset != head) ++j;
            if (j == deltas->count)
                nob_da_append(deltas, ((Delta){.offset = (int32_t)head}));
            uint8_t amount = (uint8_t)op.operand;
            deltas->items[j].delta += op.kind == OP_INC ? amount : (uint8_t)-amount;
        } else goto fail;
    }
    if (head != 0) goto fail;
    *source = 0;
    for (size_t i = 0; i < deltas->count; ++i)
        if (deltas->items[i].offset == 0) *source = deltas->items[i].delta;
    if (*source != 1 && *source != 255) goto fail;
    return true;

fail:
    nob_da_free(*deltas);
    memset(deltas, 0, sizeof(*deltas));
    return false;
}

static int compare_i32(const void *a, const void *b)
{
    int32_t x = *(const int32_t *)a, y = *(const int32_t *)b;
    return (x > y) - (x < y);
}

static bool add_offset(int32_t **items, size_t *count, size_t *capacity,
                       int64_t value)
{
    if (value < INT32_MIN || value > INT32_MAX) return false;
    int32_t offset = (int32_t)value;
    for (size_t i = 0; i < *count; ++i)
        if ((*items)[i] == offset) return true;
    if (*count >= *capacity) {
        *capacity = *capacity ? *capacity * 2 : 16;
        *items = realloc(*items, *capacity * sizeof(**items));
        if (!*items) abort();
    }
    (*items)[(*count)++] = offset;
    return true;
}

static size_t offset_index(const int32_t *offsets, size_t count, int32_t value)
{
    size_t low = 0, high = count;
    while (low < high) {
        size_t middle = low + (high - low) / 2;
        if (offsets[middle] < value) low = middle + 1;
        else high = middle;
    }
    return low < count && offsets[low] == value ? low : SIZE_MAX;
}

static uint8_t op_delta(Op op)
{
    uint8_t amount = (uint8_t)op.operand;
    return op.kind == OP_INC ? amount : (uint8_t)-amount;
}

static void matrix_multiply(uint8_t *out, const uint8_t *left,
                            const uint8_t *right, size_t width)
{
    memset(out, 0, width * width);
    for (size_t row = 0; row < width; ++row) {
        for (size_t k = 0; k < width; ++k) {
            uint8_t factor = left[row * width + k];
            if (!factor) continue;
            for (size_t column = 0; column < width; ++column)
                out[row * width + column] += factor * right[k * width + column];
        }
    }
}

static Affine_Plan *intern_affine(Affine_Plan **plans, uint32_t count,
                                  uint32_t kind, const uint8_t *matrix)
{
    size_t width = (size_t)count + 1, cells = width * width;
    for (Affine_Plan *plan = *plans; plan; plan = plan->next) {
        if (plan->count != count || plan->kind != kind) continue;
        const uint8_t *body = plan->tables + (kind == AFFINE_IF ? 0 : cells);
        if (!memcmp(body, matrix, cells)) return plan;
    }
    size_t table_count = kind == AFFINE_IF ? 1 : 256;
    Affine_Plan *plan = malloc(sizeof(*plan) + table_count * cells + 16);
    if (!plan) abort();
    plan->next = *plans;
    plan->count = count;
    plan->kind = kind;
    memset(plan->tables + table_count * cells, 0, 16);
    if (kind == AFFINE_IF) {
        memcpy(plan->tables, matrix, cells);
        atomic_init(&plan->built, true);
    } else {
        memcpy(plan->tables + cells, matrix, cells);
        atomic_init(&plan->built, false);
    }
    *plans = plan;
    return plan;
}

static void build_affine_plan(Affine_Plan *plan)
{
    if (atomic_load_explicit(&plan->built, memory_order_acquire)) return;
    if (pthread_mutex_lock(&affine_plan_mutex) != 0) abort();
    if (!atomic_load_explicit(&plan->built, memory_order_relaxed)) {
        size_t width = (size_t)plan->count + 1, cells = width * width;
        memset(plan->tables, 0, cells);
        for (size_t i = 0; i < width; ++i) plan->tables[i * width + i] = 1;
        for (size_t i = 2; i < 256; ++i)
            matrix_multiply(plan->tables + i * cells, plan->tables + cells,
                            plan->tables + (i - 1) * cells, width);
        atomic_store_explicit(&plan->built, true, memory_order_release);
    }
    if (pthread_mutex_unlock(&affine_plan_mutex) != 0) abort();
}

static bool affine_loop(Ops ops, size_t open, size_t *close, Affine **result,
                        Affine_Plan **plans)
{
    *close = ops.items[open].operand - 1;
    if (*close <= open || *close >= ops.count) return false;
    int64_t head = 0;
    int32_t *offsets = NULL;
    size_t count = 0, capacity = 0;
    if (!add_offset(&offsets, &count, &capacity, 0)) goto fail;

    for (size_t i = open + 1; i < *close; ++i) {
        Op op = ops.items[i];
        if (op.kind == OP_LEFT || op.kind == OP_RIGHT) {
            head += op.kind == OP_LEFT ? -(int64_t)op.operand : (int64_t)op.operand;
            if (head < INT32_MIN || head > INT32_MAX) goto fail;
        } else if (op.kind == OP_INC || op.kind == OP_DEC) {
            if (!add_offset(&offsets, &count, &capacity, head)) goto fail;
        } else if (op.kind == OP_JUMP_IF_ZERO) {
            size_t nested_close;
            uint8_t source;
            Deltas nested = {0};
            if (!linear_loop(ops, i, &nested_close, &nested, &source)) goto fail;
            for (size_t j = 0; j < nested.count; ++j) {
                if (!add_offset(&offsets, &count, &capacity,
                                head + nested.items[j].offset)) {
                    nob_da_free(nested);
                    goto fail;
                }
            }
            nob_da_free(nested);
            i = nested_close;
        } else goto fail;
    }
    if (head || count > 32) goto fail;

    qsort(offsets, count, sizeof(*offsets), compare_i32);
    size_t width = count + 1, cells = width * width;
    uint8_t *matrix = calloc(cells, 1);
    if (!matrix) abort();
    for (size_t i = 0; i < width; ++i) matrix[i * width + i] = 1;
    head = 0;
    for (size_t i = open + 1; i < *close; ++i) {
        Op op = ops.items[i];
        if (op.kind == OP_LEFT || op.kind == OP_RIGHT) {
            head += op.kind == OP_LEFT ? -(int64_t)op.operand : (int64_t)op.operand;
        } else if (op.kind == OP_INC || op.kind == OP_DEC) {
            size_t row = offset_index(offsets, count, (int32_t)head);
            matrix[row * width + count] += op_delta(op);
        } else {
            size_t nested_close;
            uint8_t source;
            Deltas nested = {0};
            if (!linear_loop(ops, i, &nested_close, &nested, &source)) abort();
            size_t source_row = offset_index(offsets, count, (int32_t)head);
            uint8_t saved[33];
            memcpy(saved, matrix + source_row * width, width);
            for (size_t j = 0; j < nested.count; ++j) {
                Delta delta = nested.items[j];
                if (!delta.offset || !delta.delta) continue;
                size_t row = offset_index(offsets, count,
                                          (int32_t)(head + delta.offset));
                uint8_t factor = source == 255 ? delta.delta : (uint8_t)-delta.delta;
                for (size_t column = 0; column < width; ++column)
                    matrix[row * width + column] += factor * saved[column];
            }
            memset(matrix + source_row * width, 0, width);
            nob_da_free(nested);
            i = nested_close;
        }
    }

    size_t control = offset_index(offsets, count, 0);
    uint32_t kind = 0;
    bool identity = matrix[control * width + control] == 1;
    for (size_t column = 0; column < count; ++column)
        if (column != control && matrix[control * width + column]) identity = false;
    uint8_t constant = matrix[control * width + count];
    if (identity && constant == 255) kind = AFFINE_DOWN;
    else if (identity && constant == 1) kind = AFFINE_UP;
    else {
        bool zero = true;
        for (size_t column = 0; column < width; ++column)
            if (matrix[control * width + column]) zero = false;
        if (zero) kind = AFFINE_IF;
    }
    if (!kind) {
        free(matrix);
        goto fail;
    }

    Affine *affine = malloc(sizeof(*affine) + count * sizeof(*affine->offsets));
    if (!affine) abort();
    affine->count = (uint32_t)count;
    affine->kind = kind;
    affine->control = (uint32_t)control;
    affine->plan = intern_affine(plans, (uint32_t)count, kind, matrix);
    memcpy(affine->offsets, offsets, count * sizeof(*offsets));
    free(matrix);
    free(offsets);
    *result = affine;
    return true;

fail:
    free(offsets);
    return false;
}

static bool carry_loop(Ops ops, size_t open, size_t *close, Carry *result)
{
    *close = ops.items[open].operand - 1;
    if (*close != open + 39 || *close >= ops.count) return false;
    const Op *x = ops.items + open;
#define A(i, amount) (add_amount(x[i]) == (amount))
#define M(i, amount) (move_amount(x[i]) == (amount))
#define B(i, target) (x[i].kind == OP_JUMP_IF_ZERO && x[i].operand == open + (target))
#define E(i, target) (x[i].kind == OP_JUMP_IF_NONZERO && x[i].operand == open + (target))
    int64_t to_low = move_amount(x[2]);
    int64_t low_to_scratch = move_amount(x[4]);
    int64_t flag_to_high = move_amount(x[34]);
    bool shape =
        A(1,-1) && to_low != INT64_MIN && to_low != 0 &&
        A(3,1) && low_to_scratch > 0 &&
        B(5,8) && A(6,-1) && E(7,6) && M(8,1) &&
        B(9,12) && A(10,-1) && E(11,10) && A(12,1) &&
        M(13,-(low_to_scratch+1)) &&
        B(14,24) && A(15,-1) && M(16,low_to_scratch) &&
        A(17,1) && M(18,1) && B(19,22) && A(20,-1) && E(21,20) &&
        M(22,-(low_to_scratch+1)) && E(23,15) && M(24,low_to_scratch) &&
        B(25,31) && A(26,-1) && M(27,-low_to_scratch) &&
        A(28,1) && M(29,low_to_scratch) && E(30,26) && M(31,1) &&
        B(32,38) && A(33,-1) && flag_to_high > 0 && A(35,1) &&
        M(36,-flag_to_high) && E(37,33) && E(39,1);
    int64_t low = to_low;
    int64_t scratch = low + low_to_scratch;
    int64_t flag = scratch + 1;
    int64_t high = flag + flag_to_high;
    shape = shape && M(38,-flag);
#undef A
#undef M
#undef B
#undef E
    if (!shape || low < INT32_MIN || low > INT32_MAX ||
        scratch < INT32_MIN || scratch > INT32_MAX ||
        flag < INT32_MIN || flag > INT32_MAX ||
        high < INT32_MIN || high > INT32_MAX ||
        !low || !scratch || !flag || !high ||
        low == scratch || low == flag || low == high ||
        scratch == flag || scratch == high || flag == high) return false;
    *result = (Carry){(int32_t)low, (int32_t)scratch,
                      (int32_t)flag, (int32_t)high};
    return true;
}

static void affine_exec(uint8_t *base, const Affine *affine)
{
    size_t count = affine->count, width = count + 1, cells = width * width;
    uint8_t iterations = base[affine->offsets[affine->control]];
    uint8_t state[33], next[33];
    for (size_t i = 0; i < count; ++i) state[i] = base[affine->offsets[i]];
    state[count] = 1;
    if (affine->kind == AFFINE_UP) iterations = (uint8_t)-iterations;
    if (affine->kind != AFFINE_IF && iterations != 1 &&
        !atomic_load_explicit(&affine->plan->built, memory_order_acquire))
        build_affine_plan((Affine_Plan *)affine->plan);
    const uint8_t *matrix = affine->plan->tables +
        (affine->kind == AFFINE_IF ? 0 : (size_t)iterations * cells);
    for (size_t row = 0; row < count; ++row) {
        uint32_t sum = 0;
        for (size_t column = 0; column < width; ++column)
            sum += (uint32_t)matrix[row * width + column] * state[column];
        next[row] = (uint8_t)sum;
    }
    for (size_t i = 0; i < count; ++i) base[affine->offsets[i]] = next[i];
}

#define DEFINE_AFFINE_EXEC(N) \
    static void affine_exec_##N(uint8_t *base, const Affine *affine) \
    { \
        enum { count = (N), width = (N) + 1, cells = width * width }; \
        uint8_t iterations = base[affine->offsets[affine->control]]; \
        uint8_t state[width], next[count]; \
        for (size_t i = 0; i < count; ++i) \
            state[i] = base[affine->offsets[i]]; \
        state[count] = 1; \
        if (affine->kind == AFFINE_UP) iterations = (uint8_t)-iterations; \
        if (affine->kind != AFFINE_IF && iterations != 1 && \
            !atomic_load_explicit(&affine->plan->built, memory_order_acquire)) \
            build_affine_plan((Affine_Plan *)affine->plan); \
        const uint8_t *matrix = affine->plan->tables + \
            (affine->kind == AFFINE_IF ? 0 : (size_t)iterations * cells); \
        for (size_t row = 0; row < count; ++row) { \
            uint32_t sum = 0; \
            for (size_t column = 0; column < width; ++column) \
                sum += (uint32_t)matrix[row * width + column] * state[column]; \
            next[row] = (uint8_t)sum; \
        } \
        for (size_t i = 0; i < count; ++i) \
            base[affine->offsets[i]] = next[i]; \
    }

DEFINE_AFFINE_EXEC(3)
DEFINE_AFFINE_EXEC(4)
DEFINE_AFFINE_EXEC(28)
DEFINE_AFFINE_EXEC(29)
#undef DEFINE_AFFINE_EXEC

static void affine_exec_4_down(uint8_t *base, const Affine *affine)
{
    enum { count = 4, width = 5, cells = width * width };
    uint8_t iterations = base[affine->offsets[affine->control]];
    uint8_t state[width], next[count];
    for (size_t i = 0; i < count; ++i)
        state[i] = base[affine->offsets[i]];
    state[count] = 1;
    if (iterations != 1 &&
        !atomic_load_explicit(&affine->plan->built, memory_order_acquire))
        build_affine_plan((Affine_Plan *)affine->plan);
    const uint8_t *matrix = affine->plan->tables + (size_t)iterations * cells;
    for (size_t row = 0; row < count; ++row) {
        uint32_t sum = 0;
        for (size_t column = 0; column < width; ++column)
            sum += (uint32_t)matrix[row * width + column] * state[column];
        next[row] = (uint8_t)sum;
    }
    for (size_t i = 0; i < count; ++i)
        base[affine->offsets[i]] = next[i];
}

static void affine_exec_14(uint8_t *base, const Affine *affine)
{
    enum { count = 14, width = 15, cells = width * width };
    uint8_t iterations = base[affine->offsets[affine->control]];
    uint8_t state[16] = {0}, next[count];
    for (size_t i = 0; i < count; ++i)
        state[i] = base[affine->offsets[i]];
    state[count] = 1;
    if (affine->kind == AFFINE_UP) iterations = (uint8_t)-iterations;
    if (affine->kind != AFFINE_IF && iterations != 1 &&
        !atomic_load_explicit(&affine->plan->built, memory_order_acquire))
        build_affine_plan((Affine_Plan *)affine->plan);
    const uint8_t *matrix = affine->plan->tables +
        (affine->kind == AFFINE_IF ? 0 : (size_t)iterations * cells);
    __m128i zero = _mm_setzero_si128();
    __m128i state_bytes = _mm_loadu_si128((const __m128i *)state);
    __m128i state_low = _mm_unpacklo_epi8(state_bytes, zero);
    __m128i state_high = _mm_unpackhi_epi8(state_bytes, zero);
    for (size_t row = 0; row < count; ++row) {
        __m128i coefficients = _mm_loadu_si128(
            (const __m128i *)(matrix + row * width));
        __m128i low = _mm_mullo_epi16(
            _mm_unpacklo_epi8(coefficients, zero), state_low);
        __m128i high = _mm_mullo_epi16(
            _mm_unpackhi_epi8(coefficients, zero), state_high);
        __m128i sum = _mm_add_epi16(low, high);
        sum = _mm_add_epi16(sum, _mm_srli_si128(sum, 8));
        sum = _mm_add_epi16(sum, _mm_srli_si128(sum, 4));
        sum = _mm_add_epi16(sum, _mm_srli_si128(sum, 2));
        next[row] = (uint8_t)_mm_cvtsi128_si32(sum);
    }
    for (size_t i = 0; i < count; ++i)
        base[affine->offsets[i]] = next[i];
}

__attribute__((target("avx2")))
static void affine_exec_29_avx2(uint8_t *base, const Affine *affine)
{
    enum { count = 29, width = 30, cells = width * width };
    uint8_t iterations = base[affine->offsets[affine->control]];
    uint8_t state[32] = {0}, next[count];
    for (size_t i = 0; i < count; ++i)
        state[i] = base[affine->offsets[i]];
    state[count] = 1;
    if (affine->kind == AFFINE_UP) iterations = (uint8_t)-iterations;
    if (affine->kind != AFFINE_IF && iterations != 1 &&
        !atomic_load_explicit(&affine->plan->built, memory_order_acquire))
        build_affine_plan((Affine_Plan *)affine->plan);
    const uint8_t *matrix = affine->plan->tables +
        (affine->kind == AFFINE_IF ? 0 : (size_t)iterations * cells);
    __m256i zero = _mm256_setzero_si256();
    __m256i state_bytes = _mm256_loadu_si256((const __m256i *)state);
    __m256i state_low = _mm256_unpacklo_epi8(state_bytes, zero);
    __m256i state_high = _mm256_unpackhi_epi8(state_bytes, zero);
    for (size_t row = 0; row < count; ++row) {
        __m256i coefficients = _mm256_loadu_si256(
            (const __m256i *)(matrix + row * width));
        __m256i products = _mm256_add_epi32(
            _mm256_madd_epi16(
                _mm256_unpacklo_epi8(coefficients, zero), state_low),
            _mm256_madd_epi16(
                _mm256_unpackhi_epi8(coefficients, zero), state_high));
        __m128i sum = _mm_add_epi32(
            _mm256_castsi256_si128(products),
            _mm256_extracti128_si256(products, 1));
        sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));
        sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 4));
        next[row] = (uint8_t)_mm_cvtsi128_si32(sum);
    }
    for (size_t i = 0; i < count; ++i)
        base[affine->offsets[i]] = next[i];
}

static void fixed_sqrt_exec(uint8_t *cells)
{
    uint32_t input;
    memcpy(&input, cells + 4, sizeof(input));
    uint64_t radicand = (uint64_t)input << 16;
    // This 32-bit-aligned radicand is exact in binary64; its root is below
    // 2^24, where sqrt rounding is far too small to cross an integer boundary.
    uint32_t root = (uint32_t)__builtin_sqrt((double)radicand);
    uint64_t square = (uint64_t)root * root;
    uint32_t remainder = (uint32_t)(radicand - square);
    uint32_t trial = root & 1 ? 0 : (root << 1) + 1;
    memcpy(cells, &root, sizeof(root));
    memset(cells + 4, 0, 4);
    memcpy(cells + 8, &remainder, sizeof(remainder));
    memset(cells + 12, 0, 4);
    memcpy(cells + 16, &trial, sizeof(trial));
    memset(cells + 20, 0, 8);
}

static void fixed_integer_exec(uint8_t *cells)
{
    int32_t input;
    memcpy(&input, cells + 4, sizeof(input));
    uint32_t magnitude = input < 0 ? -(uint32_t)input : (uint32_t)input;
    uint32_t result = magnitude & UINT32_C(0xffff0000);
    if (input < 0) result = -result;
    memcpy(cells, &result, sizeof(result));
    memset(cells + 4, 0, 7);
    memset(cells + 12, 0, 8);
}

static void input_repeat(uint8_t *cell, uint32_t count)
{
    while (count--) {
        int c = getchar_unlocked();
        if (c != EOF) *cell = (uint8_t)c;
    }
}

static void input_repeat_locked(uint8_t *cell, uint32_t count)
{
    while (count--) {
        int c = getchar();
        if (c != EOF) *cell = (uint8_t)c;
    }
}

static void output_repeat(uint8_t value, uint32_t count)
{
    uint8_t buffer[256];
    memset(buffer, value, sizeof(buffer));
    while (count >= sizeof(buffer)) {
        fwrite(buffer, sizeof(buffer), 1, stdout);
        count -= sizeof(buffer);
    }
    if (count) fwrite(buffer, count, 1, stdout);
}

static void emit_modrm_rdi(Nob_String_Builder *sb, uint8_t reg, int32_t offset)
{
    if (!offset) {
        nob_da_append(sb, (uint8_t)(0x07 | reg << 3));
    } else if (offset >= INT8_MIN && offset <= INT8_MAX) {
        nob_da_append(sb, (uint8_t)(0x47 | reg << 3));
        nob_da_append(sb, (uint8_t)offset);
    } else {
        nob_da_append(sb, (uint8_t)(0x87 | reg << 3));
        nob_da_append_many(sb, &offset, sizeof(offset));
    }
}

static void emit_zero(Nob_String_Builder *sb, int32_t offset)
{
    nob_da_append(sb, 0xc6);
    emit_modrm_rdi(sb, 0, offset);
    nob_da_append(sb, 0);
}

static void emit_zero_dword(Nob_String_Builder *sb, int32_t offset)
{
    nob_da_append(sb, 0xc7);
    emit_modrm_rdi(sb, 0, offset);
    uint32_t zero = 0;
    nob_da_append_many(sb, &zero, sizeof(zero));
}

static void emit_add_byte(Nob_String_Builder *sb, bool subtract,
                          int32_t offset, uint8_t reg)
{
    nob_da_append(sb, subtract ? 0x28 : 0x00);
    emit_modrm_rdi(sb, reg, offset);
}

static void emit_store_byte(Nob_String_Builder *sb, uint8_t reg, int32_t offset)
{
    nob_da_append(sb, 0x88);
    emit_modrm_rdi(sb, reg, offset);
}

static void emit_head_adjust(Nob_String_Builder *sb, int32_t amount)
{
    if (!amount) return;
    if (amount >= INT8_MIN && amount <= INT8_MAX) {
        nob_da_append_many(sb, "\x48\x83\xc7", 3);
        nob_da_append(sb, (uint8_t)amount);
    } else {
        nob_da_append_many(sb, amount > 0 ? "\x48\x81\xc7" : "\x48\x81\xef", 3);
        uint32_t distance = amount > 0 ? (uint32_t)amount : (uint32_t)-amount;
        nob_da_append_many(sb, &distance, sizeof(distance));
    }
}

static void emit_scan(Nob_String_Builder *sb, int32_t stride)
{
    nob_da_append_many(sb, "\x80\x3f\x00\x74\x00", 5); // cmp [rdi],0; jz done
    size_t done_patch = sb->count - 1;
    size_t body = sb->count;
    emit_head_adjust(sb, stride);
    nob_da_append_many(sb, "\x80\x3f\x00\x75\x00", 5); // cmp [rdi],0; jnz body
    size_t back_patch = sb->count - 1;
    ptrdiff_t back = (ptrdiff_t)body - (ptrdiff_t)sb->count;
    ptrdiff_t done = (ptrdiff_t)sb->count - (ptrdiff_t)(done_patch + 1);
    assert(back >= INT8_MIN && back <= INT8_MAX);
    assert(done >= INT8_MIN && done <= INT8_MAX);
    sb->items[back_patch] = (uint8_t)back;
    sb->items[done_patch] = (uint8_t)done;
}

static void emit_copy(Nob_String_Builder *sb, Copy copy, int32_t base)
{
    nob_da_append_many(sb, "\x0f\xb6", 2);
    emit_modrm_rdi(sb, 0, base + copy.source);
    emit_store_byte(sb, 0, base);
    emit_zero(sb, base + (copy.destructive ? copy.source : copy.temporary));
}

static void emit_carry(Nob_String_Builder *sb, Carry carry, int32_t base)
{
    nob_da_append_many(sb, "\x0f\xb6", 2);
    emit_modrm_rdi(sb, 0, base);
    nob_da_append_many(sb, "\x85\xc0\x74", 3);
    size_t patch = sb->count;
    nob_da_append(sb, 0);
    size_t source = sb->count;
    nob_da_append_many(sb, "\x0f\xb6", 2);
    emit_modrm_rdi(sb, 1, base + carry.low);
    nob_sb_append_cstr(sb, "\x01\xc1");
    emit_store_byte(sb, 1, base + carry.low);
    nob_da_append_many(sb, "\xc1\xe9\x08", 3);
    emit_add_byte(sb, false, base + carry.high, 1);
    emit_zero(sb, base + carry.scratch);
    emit_zero(sb, base + carry.flag);
    emit_zero(sb, base);
    size_t distance = sb->count - source;
    assert(distance <= INT8_MAX);
    sb->items[patch] = (uint8_t)distance;
}

static void emit_increment_carry(Nob_String_Builder *sb,
                                 Increment_Carry increment, int32_t base)
{
    nob_da_append(sb, 0x80);
    emit_modrm_rdi(sb, 0, base);
    nob_da_append(sb, 1);
    nob_da_append_many(sb, "\x75\x00", 2);
    size_t patch = sb->count - 1, source = sb->count;
    nob_da_append(sb, 0x80);
    emit_modrm_rdi(sb, 0, base + increment.carry);
    nob_da_append(sb, 1);
    size_t distance = sb->count - source;
    assert(distance <= INT8_MAX);
    sb->items[patch] = (uint8_t)distance;
    emit_zero(sb, base + increment.scratch);
    emit_zero(sb, base + increment.flag);
}

static void emit_multi_add(Nob_String_Builder *sb, Multi_Add add, int32_t base)
{
    nob_da_append_many(sb, "\x0f\xb6", 2); // movzx eax, byte [source]
    emit_modrm_rdi(sb, 0, base);
    nob_sb_append_cstr(sb, "\x85\xc0\x0f\x84"); // test eax, eax; jz done
    size_t patch = sb->count;
    uint32_t zero = 0;
    nob_da_append_many(sb, &zero, sizeof(zero));
    size_t source = sb->count;

    emit_add_byte(sb, false, base + add.destinations[0], 0);
    for (uint32_t i = 1; i < add.destination_count; ++i) {
        nob_da_append(sb, 0x80); // adc byte [destination], 0
        emit_modrm_rdi(sb, 2, base + add.destinations[i]);
        nob_da_append(sb, 0);
    }
    emit_zero(sb, base);
    for (uint32_t i = 0; i < add.zero_count; ++i)
        emit_zero(sb, base + add.zeros[i]);

    int32_t relative = (int32_t)(sb->count - source);
    memcpy(&sb->items[patch], &relative, sizeof(relative));
}

static void emit_multi_multiply(Nob_String_Builder *sb,
                                Multi_Multiply multiply, int32_t base)
{
    nob_da_append_many(sb, "\x0f\xb6", 2); // movzx eax, byte [source]
    emit_modrm_rdi(sb, 0, base);
    nob_sb_append_cstr(sb, "\x85\xc0\x0f\x84"); // test eax, eax; jz done
    size_t patch = sb->count;
    uint32_t zero = 0;
    nob_da_append_many(sb, &zero, sizeof(zero));
    size_t source = sb->count;
    nob_da_append_many(sb, "\x0f\xb6", 2); // movzx ecx, byte [multiplier]
    emit_modrm_rdi(sb, 1, base + multiply.multiplier);
    nob_da_append_many(sb, "\x0f\xaf\xc1", 3); // imul eax, ecx
    for (uint32_t i = 0; i < multiply.destination_count; ++i) {
        nob_da_append_many(sb, "\x0f\xb6", 2); // movzx ecx, byte [destination]
        emit_modrm_rdi(sb, 1, base + multiply.destinations[i]);
        nob_sb_append_cstr(sb, "\x01\xc8"); // add eax, ecx
        emit_store_byte(sb, 0, base + multiply.destinations[i]);
        nob_da_append_many(sb, "\xc1\xe8\x08", 3); // shr eax, 8
    }
    emit_zero(sb, base);
    for (uint32_t i = 0; i < multiply.zero_count; ++i)
        emit_zero(sb, base + multiply.zeros[i]);
    int32_t relative = (int32_t)(sb->count - source);
    memcpy(&sb->items[patch], &relative, sizeof(relative));
}

static void emit_byte_compare(Nob_String_Builder *sb, Byte_Compare compare,
                              int32_t base)
{
    nob_da_append_many(sb, "\x0f\xb6", 2); // movzx eax, byte [x]
    emit_modrm_rdi(sb, 0, base);
    nob_da_append_many(sb, "\x85\xc0\x0f\x84", 4); // test eax, eax; jz done
    size_t done_patch = sb->count;
    uint32_t zero = 0;
    nob_da_append_many(sb, &zero, sizeof(zero));
    size_t done_source = sb->count;

    nob_da_append_many(sb, "\x0f\xb6", 2); // movzx ecx, byte [y]
    emit_modrm_rdi(sb, 1, base + compare.y);
    nob_da_append_many(sb, "\x39\xc8\x0f\x86", 4); // cmp eax, ecx; jbe x_le_y
    size_t less_patch = sb->count;
    nob_da_append_many(sb, &zero, sizeof(zero));
    size_t less_source = sb->count;

    nob_da_append(sb, 0xc6); // mov byte [greater], 1
    emit_modrm_rdi(sb, 0, base + compare.greater);
    nob_da_append(sb, 1);
    emit_zero(sb, base + compare.y);
    nob_da_append(sb, 0xe9); // jmp cleared
    size_t clear_patch = sb->count;
    nob_da_append_many(sb, &zero, sizeof(zero));
    size_t clear_source = sb->count;

    int32_t relative = (int32_t)(sb->count - less_source);
    memcpy(&sb->items[less_patch], &relative, sizeof(relative));
    nob_sb_append_cstr(sb, "\x29\xc1"); // sub ecx, eax
    emit_store_byte(sb, 1, base + compare.y);

    relative = (int32_t)(sb->count - clear_source);
    memcpy(&sb->items[clear_patch], &relative, sizeof(relative));
    emit_zero(sb, base);
    emit_zero(sb, base + compare.copy);
    emit_zero(sb, base + compare.temporary);
    emit_zero(sb, base + compare.flag);
    emit_zero(sb, base + compare.guard);

    relative = (int32_t)(sb->count - done_source);
    memcpy(&sb->items[done_patch], &relative, sizeof(relative));
}

static void emit_division_digit(Nob_String_Builder *sb, int32_t base)
{
    nob_da_append(sb, 0x80); // cmp byte [control], 0
    emit_modrm_rdi(sb, 7, base);
    nob_da_append(sb, 0);
    nob_sb_append_cstr(sb, "\x0f\x84"); // jz done
    size_t done_patch = sb->count;
    uint32_t zero = 0;
    nob_da_append_many(sb, &zero, sizeof(zero));
    size_t done_source = sb->count;

    nob_da_append(sb, 0x8b); // mov ecx, dword [divisor]
    emit_modrm_rdi(sb, 1, base - 19);
    nob_sb_append_cstr(sb, "\x85\xc9\x0f\x84"); // test ecx, ecx; jz clear
    size_t clear_patch = sb->count;
    nob_da_append_many(sb, &zero, sizeof(zero));
    size_t clear_source = sb->count;

    nob_da_append_many(sb, "\x48\x8b", 2); // mov rax, qword [remainder]
    emit_modrm_rdi(sb, 0, base - 15);
    nob_da_append_many(sb, "\x48\xc1\xe0\x18\x48\xc1\xe8\x18", 8); // mask to 40 bits
    nob_da_append_many(sb, "\x31\xd2\x48\xf7\xf1", 5); // xor edx, edx; div rcx
    emit_add_byte(sb, false, base - 27, 0); // quotient += al
    nob_da_append(sb, 0x89); // mov dword [remainder], edx
    emit_modrm_rdi(sb, 2, base - 15);
    emit_zero(sb, base - 11);

    int32_t relative = (int32_t)(sb->count - clear_source);
    memcpy(&sb->items[clear_patch], &relative, sizeof(relative));
    for (int32_t offset = -9; offset <= 0; ++offset)
        emit_zero(sb, base + offset);

    relative = (int32_t)(sb->count - done_source);
    memcpy(&sb->items[done_patch], &relative, sizeof(relative));
}

static void emit_subdigit(Nob_String_Builder *sb, Subdigit subdigit,
                          int32_t base)
{
    nob_da_append_many(sb, "\x0f\xb7", 2); // movzx eax, word [number]
    emit_modrm_rdi(sb, 0, base - 4);
    nob_sb_append_cstr(sb, "\x31\xd2\xb9"); // xor edx, edx; mov ecx, divisor
    uint32_t divisor = subdigit.divisor;
    nob_da_append_many(sb, &divisor, sizeof(divisor));
    nob_sb_append_cstr(sb, "\xf7\xf1"); // div ecx
    emit_add_byte(sb, false, base + subdigit.digit, 0);
    nob_da_append_many(sb, "\x66\x89", 2); // mov word [number], dx
    emit_modrm_rdi(sb, 2, base - 4);

    nob_da_append(sb, 0xc6); // mov byte [divisor], immediate
    emit_modrm_rdi(sb, 0, base);
    nob_da_append(sb, subdigit.divisor);
    for (int32_t offset = 1; offset <= 5; ++offset)
        emit_zero(sb, base + offset);
    for (int32_t offset = 8; offset <= 19; ++offset)
        emit_zero(sb, base + offset);
}

static void emit_modulo_256(Nob_String_Builder *sb, Modulo_256 modulo,
                            int32_t base)
{
    nob_da_append(sb, 0x80); // cmp byte [control], 0
    emit_modrm_rdi(sb, 7, base);
    nob_da_append(sb, 0);
    nob_sb_append_cstr(sb, "\x0f\x84"); // jz done
    size_t done_patch = sb->count;
    uint32_t zero = 0;
    nob_da_append_many(sb, &zero, sizeof(zero));
    size_t done_source = sb->count;

    int32_t state = base + modulo.state;
    emit_zero(sb, state + 3); // Q16.16 value modulo 256
    for (int32_t offset = 4; offset <= 15; ++offset)
        emit_zero(sb, state + offset);
    emit_zero(sb, state + 24);
    for (int32_t offset = 28; offset <= 53; ++offset)
        emit_zero(sb, state + offset);
    emit_zero(sb, state + 55);
    for (int32_t offset = -5; offset <= 0; ++offset)
        emit_zero(sb, base + offset);

    int32_t relative = (int32_t)(sb->count - done_source);
    memcpy(&sb->items[done_patch], &relative, sizeof(relative));
}

static void emit_complement(Nob_String_Builder *sb, Complement complement,
                            int32_t base)
{
    nob_da_append(sb, 0xf6); // not byte [target]
    emit_modrm_rdi(sb, 2, base + complement.target);
    emit_zero(sb, base);
}

static void emit_truth(Nob_String_Builder *sb, Truth truth, int32_t base)
{
    nob_da_append(sb, 0x80); // cmp byte [source], 0
    emit_modrm_rdi(sb, 7, base + truth.source);
    nob_da_append(sb, 0);
    nob_da_append_many(sb, "\x0f\x95\xc0", 3); // setne al
    emit_store_byte(sb, 0, base);
    emit_zero(sb, base + truth.source);
}

static void emit_linear(Nob_String_Builder *sb, Deltas deltas, uint8_t source,
                        int32_t base)
{
    nob_da_append_many(sb, "\x0f\xb6", 2); // movzx eax, byte [rdi+base]
    emit_modrm_rdi(sb, 0, base);
    for (size_t i = 0; i < deltas.count; ++i) {
        Delta delta = deltas.items[i];
        if (!delta.offset || !delta.delta) continue;
        uint8_t factor = source == 255 ? delta.delta : (uint8_t)-delta.delta;
        uint8_t reg = 0;
        if (factor != 1 && factor != 255) {
            nob_sb_append_cstr(sb, "\x89\xc1\x69\xc9"); // mov ecx,eax; imul ecx,factor
            uint32_t multiplier = factor;
            nob_da_append_many(sb, &multiplier, sizeof(multiplier));
            reg = 1;
        }
        nob_da_append(sb, factor == 255 ? 0x28 : 0x00); // sub/add [target], al/cl
        int64_t target = (int64_t)base + delta.offset;
        assert(target >= INT32_MIN && target <= INT32_MAX);
        emit_modrm_rdi(sb, reg, (int32_t)target);
    }
    emit_zero(sb, base);
}

static void emit_affine_term(Nob_String_Builder *sb, uint8_t factor,
                             uint8_t source)
{
    if (!factor) return;
    uint8_t source_modrm = source ? 0xca : 0xc2; // add edx, ecx/eax
    if (factor == 1) {
        nob_da_append(sb, 0x01);
        nob_da_append(sb, source_modrm);
    } else if (factor == 255) {
        nob_da_append(sb, 0x29);
        nob_da_append(sb, source_modrm);
    } else {
        nob_da_append(sb, 0x69);
        nob_da_append(sb, (uint8_t)(source ? 0xf1 : 0xf0)); // imul esi, ecx/eax
        uint32_t multiplier = factor;
        nob_da_append_many(sb, &multiplier, sizeof(multiplier));
        nob_da_append_many(sb, "\x01\xf2", 2); // add edx, esi
    }
}

static void emit_affine_4_once(Nob_String_Builder *sb, const Affine *affine)
{
    enum { count = 4, width = 5, cells = width * width };
    const uint8_t *matrix = affine->plan->tables + cells;
    bool used[count];
    memset(used, 0, sizeof(used));
    for (size_t row = 0; row < count; ++row)
        for (size_t column = 0; column < count; ++column)
            used[column] |= matrix[row * width + column] != 0;

    for (size_t column = 0; column < count; ++column) {
        if (!used[column]) continue;
        if (column >= 2) nob_da_append(sb, 0x44); // REX.R for r8d/r9d
        nob_da_append_many(sb, "\x0f\xb6", 2); // movzx source, byte [cell]
        emit_modrm_rdi(sb, (uint8_t)(column & 1), affine->offsets[column]);
    }
    static const uint8_t add[][3] = {
        {0x01, 0xc2}, {0x01, 0xca}, {0x44, 0x01, 0xc2}, {0x44, 0x01, 0xca},
    };
    static const uint8_t sub[][3] = {
        {0x29, 0xc2}, {0x29, 0xca}, {0x44, 0x29, 0xc2}, {0x44, 0x29, 0xca},
    };
    static const uint8_t lengths[] = {2, 2, 3, 3};
    static const uint8_t multiply[][3] = {
        {0x69, 0xf0}, {0x69, 0xf1}, {0x41, 0x69, 0xf0}, {0x41, 0x69, 0xf1},
    };
    for (size_t row = 0; row < count; ++row) {
        nob_da_append(sb, 0xba); // mov edx, constant
        uint32_t constant = matrix[row * width + count];
        nob_da_append_many(sb, &constant, sizeof(constant));
        for (size_t column = 0; column < count; ++column) {
            uint8_t factor = matrix[row * width + column];
            if (!factor) continue;
            if (factor == 1)
                nob_da_append_many(sb, add[column], lengths[column]);
            else if (factor == 255)
                nob_da_append_many(sb, sub[column], lengths[column]);
            else {
                nob_da_append_many(sb, multiply[column], lengths[column]);
                uint32_t multiplier = factor;
                nob_da_append_many(sb, &multiplier, sizeof(multiplier));
                nob_da_append_many(sb, "\x01\xf2", 2); // add edx, esi
            }
        }
        emit_store_byte(sb, 2, affine->offsets[row]);
    }
}

static bool emit_small_affine_if(Nob_String_Builder *sb, const Affine *affine,
                                 int32_t base)
{
    if (affine->kind != AFFINE_IF || affine->count != 2) return false;
    size_t width = 3;
    const uint8_t *matrix = affine->plan->tables;
    int64_t offsets[2] = {(int64_t)base + affine->offsets[0],
                          (int64_t)base + affine->offsets[1]};
    if (offsets[0] < INT32_MIN || offsets[0] > INT32_MAX ||
        offsets[1] < INT32_MIN || offsets[1] > INT32_MAX) return false;
    int32_t control = (int32_t)offsets[affine->control];
    nob_da_append(sb, 0x80); // cmp byte [control], 0
    emit_modrm_rdi(sb, 7, control);
    nob_da_append(sb, 0);
    nob_sb_append_cstr(sb, "\x0f\x84"); // jz done
    size_t done_patch = sb->count;
    uint32_t zero = 0;
    nob_da_append_many(sb, &zero, sizeof(zero));
    size_t body = sb->count;

    nob_da_append_many(sb, "\x0f\xb6", 2); // movzx eax, byte [cell 0]
    emit_modrm_rdi(sb, 0, (int32_t)offsets[0]);
    nob_da_append_many(sb, "\x0f\xb6", 2); // movzx ecx, byte [cell 1]
    emit_modrm_rdi(sb, 1, (int32_t)offsets[1]);
    for (size_t row = 0; row < 2; ++row) {
        nob_da_append(sb, 0xba); // mov edx, constant
        uint32_t constant = matrix[row * width + 2];
        nob_da_append_many(sb, &constant, sizeof(constant));
        emit_affine_term(sb, matrix[row * width], 0);
        emit_affine_term(sb, matrix[row * width + 1], 1);
        emit_store_byte(sb, 2, (int32_t)offsets[row]);
    }
    int32_t relative = (int32_t)(sb->count - body);
    memcpy(&sb->items[done_patch], &relative, sizeof(relative));
    return true;
}

static bool emit_simple_affine(Nob_String_Builder *sb, const Affine *affine,
                               int32_t base)
{
    if (affine->kind != AFFINE_DOWN && affine->kind != AFFINE_UP) return false;
    size_t count = affine->count, width = count + 1, cells = width * width;
    const uint8_t *matrix = affine->plan->tables + cells;
    size_t control = affine->control;
    for (size_t row = 0; row < count; ++row) {
        if (row == control) continue;
        for (size_t column = 0; column < count; ++column) {
            uint8_t expected = column == row ? matrix[row * width + row] : 0;
            if (matrix[row * width + column] != expected) return false;
        }
        if (matrix[row * width + row] > 1) return false;
    }

    int64_t control_offset = (int64_t)base + affine->offsets[control];
    if (control_offset < INT32_MIN || control_offset > INT32_MAX) return false;
    nob_da_append_many(sb, "\x0f\xb6", 2);
    emit_modrm_rdi(sb, 0, (int32_t)control_offset);
    nob_da_append_many(sb, "\x85\xc0\x0f\x84", 4);
    size_t patch = sb->count;
    uint32_t zero = 0;
    nob_da_append_many(sb, &zero, sizeof(zero));
    size_t source = sb->count;
    if (affine->kind == AFFINE_UP) nob_sb_append_cstr(sb, "\xf7\xd8");

    for (size_t row = 0; row < count; ++row) {
        if (row == control) continue;
        int64_t target = (int64_t)base + affine->offsets[row];
        if (target < INT32_MIN || target > INT32_MAX) return false;
        uint8_t self = matrix[row * width + row];
        uint8_t constant = matrix[row * width + count];
        if (!self) {
            nob_da_append(sb, 0xc6);
            emit_modrm_rdi(sb, 0, (int32_t)target);
            nob_da_append(sb, constant);
        } else if (constant == 1) {
            emit_add_byte(sb, false, (int32_t)target, 0);
        } else if (constant == 255) {
            emit_add_byte(sb, true, (int32_t)target, 0);
        } else if (constant) {
            nob_sb_append_cstr(sb, "\x89\xc1\x69\xc9");
            uint32_t multiplier = constant;
            nob_da_append_many(sb, &multiplier, sizeof(multiplier));
            emit_add_byte(sb, false, (int32_t)target, 1);
        }
    }
    emit_zero(sb, (int32_t)control_offset);
    int32_t relative = (int32_t)(sb->count - source);
    memcpy(&sb->items[patch], &relative, sizeof(relative));
    return true;
}

static uintptr_t fnaddr(const void *p, size_t n)
{
    uintptr_t result = 0;
    assert(n <= sizeof(result));
    memcpy(&result, p, n);
    return result;
}

static void emit_call(Nob_String_Builder *sb, uintptr_t address)
{
    nob_sb_append_cstr(sb, "\x48\xb8");
    nob_da_append_many(sb, &address, sizeof(address));
    nob_sb_append_cstr(sb, "\xff\xd0");
}

static void emit_fixed_multiply_call(Nob_String_Builder *sb, int32_t base,
                                     Addrs *calls)
{
    int32_t destination = base - 27;
    if (destination) {
        nob_sb_append_cstr(sb, "\x48\x8d\xb7"); // lea rsi, [rdi+destination]
        nob_da_append_many(sb, &destination, sizeof(destination));
    } else nob_sb_append_cstr(sb, "\x48\x89\xfe"); // mov rsi, rdi
    nob_da_append(sb, 0xe8); // call multiply stub
    nob_da_append(calls, sb->count);
    uint32_t zero = 0;
    nob_da_append_many(sb, &zero, sizeof(zero));
}

static void emit_fixed_sqrt_call(Nob_String_Builder *sb, int32_t base)
{
    nob_da_append(sb, 0x57); // push rdi
    int32_t destination = base - 27;
    if (destination) {
        nob_sb_append_cstr(sb, "\x48\x8d\xbf"); // lea rdi, [rdi+destination]
        nob_da_append_many(sb, &destination, sizeof(destination));
    }
    void (*helper)(uint8_t *) = fixed_sqrt_exec;
    emit_call(sb, fnaddr(&helper, sizeof(helper)));
    nob_da_append(sb, 0x5f); // pop rdi
}

static void emit_fixed_compare_call(Nob_String_Builder *sb,
                                    Fixed_Compare compare, int32_t base,
                                    Addrs compare_calls[])
{
    int32_t destination = base + compare.destination;
    if (destination) {
        nob_sb_append_cstr(sb, "\x48\x8d\xb7"); // lea rsi, [rdi+destination]
        nob_da_append_many(sb, &destination, sizeof(destination));
    } else nob_sb_append_cstr(sb, "\x48\x89\xfe"); // mov rsi, rdi
    assert(compare.comparison < 5);
    nob_da_append(sb, 0xe8); // call comparison stub
    nob_da_append(&compare_calls[compare.comparison], sb->count);
    uint32_t zero = 0;
    nob_da_append_many(sb, &zero, sizeof(zero));
}

static void emit_fixed_add_call(Nob_String_Builder *sb, Fixed_Add add,
                                int32_t base)
{
    int32_t destination = base + add.destination;
    nob_da_append(sb, 0x8b); // mov eax, dword [x]
    emit_modrm_rdi(sb, 0, destination + 4);
    nob_da_append(sb, add.subtract ? 0x2b : 0x03); // sub/add eax, dword [y]
    emit_modrm_rdi(sb, 0, destination + 8);
    nob_da_append(sb, 0x89); // mov dword [result], eax
    emit_modrm_rdi(sb, 0, destination);
    emit_zero_dword(sb, destination + 4);
    emit_zero_dword(sb, destination + 8);
    emit_zero_dword(sb, destination + 12);
}

static void emit_fixed_divide_call(Nob_String_Builder *sb, int32_t base,
                                   Addrs *calls)
{
    int32_t destination = base - 20;
    if (destination) {
        nob_sb_append_cstr(sb, "\x48\x8d\xb7"); // lea rsi, [rdi+destination]
        nob_da_append_many(sb, &destination, sizeof(destination));
    } else nob_sb_append_cstr(sb, "\x48\x89\xfe"); // mov rsi, rdi
    nob_da_append(sb, 0xe8); // call divide stub
    nob_da_append(calls, sb->count);
    uint32_t zero = 0;
    nob_da_append_many(sb, &zero, sizeof(zero));
}

static void emit_fixed_integer_call(Nob_String_Builder *sb, int32_t base)
{
    nob_da_append(sb, 0x57); // push rdi
    int32_t destination = base - 12;
    if (destination) {
        nob_sb_append_cstr(sb, "\x48\x8d\xbf"); // lea rdi, [rdi+destination]
        nob_da_append_many(sb, &destination, sizeof(destination));
    }
    void (*helper)(uint8_t *) = fixed_integer_exec;
    emit_call(sb, fnaddr(&helper, sizeof(helper)));
    nob_da_append(sb, 0x5f); // pop rdi
}

static void emit_affine_call(Nob_String_Builder *sb, const Affine *affine,
                             int32_t base)
{
    nob_da_append(sb, 0x57); // push rdi
    if (base) {
        nob_sb_append_cstr(sb, "\x48\x8d\xbf"); // lea rdi, [rdi+base]
        nob_da_append_many(sb, &base, sizeof(base));
    }
    nob_da_append(sb, 0x80); // cmp byte [control], 0
    emit_modrm_rdi(sb, 7, affine->offsets[affine->control]);
    nob_da_append(sb, 0);
    nob_sb_append_cstr(sb, "\x0f\x84"); // jz done
    size_t done_patch = sb->count;
    uint32_t zero = 0;
    nob_da_append_many(sb, &zero, sizeof(zero));
    size_t once_done_patch = 0;
    if (affine->count == 4 && affine->kind == AFFINE_DOWN) {
        nob_da_append(sb, 0x80); // cmp byte [control], 1
        emit_modrm_rdi(sb, 7, affine->offsets[affine->control]);
        nob_sb_append_cstr(sb, "\x01\x0f\x85"); // jne fallback
        size_t fallback_patch = sb->count;
        nob_da_append_many(sb, &zero, sizeof(zero));
        emit_affine_4_once(sb, affine);
        nob_da_append(sb, 0xe9); // jmp done
        once_done_patch = sb->count;
        nob_da_append_many(sb, &zero, sizeof(zero));
        int32_t relative = (int32_t)(sb->count - (fallback_patch + 4));
        memcpy(sb->items + fallback_patch, &relative, sizeof(relative));
    }
    nob_sb_append_cstr(sb, "\x48\xbe"); // movabs rsi, affine
    uintptr_t descriptor = (uintptr_t)affine;
    nob_da_append_many(sb, &descriptor, sizeof(descriptor));
    void (*helper)(uint8_t *, const Affine *);
    switch (affine->count) {
        case 3: helper = affine_exec_3; break;
        case 4: helper = affine->kind == AFFINE_DOWN ?
            affine_exec_4_down : affine_exec_4; break;
        case 14: helper = affine_exec_14; break;
        case 28: helper = affine_exec_28; break;
        case 29:
            helper = __builtin_cpu_supports("avx2") ?
                affine_exec_29_avx2 : affine_exec_29;
            break;
        default: helper = affine_exec; break;
    }
    emit_call(sb, fnaddr(&helper, sizeof(helper)));
    if (once_done_patch) {
        int32_t relative = (int32_t)(sb->count - (once_done_patch + 4));
        memcpy(sb->items + once_done_patch, &relative, sizeof(relative));
    }
    int32_t relative = (int32_t)(sb->count - (done_patch + 4));
    memcpy(sb->items + done_patch, &relative, sizeof(relative));
    nob_da_append(sb, 0x5f); // pop rdi
}

bool jit_compile(Ops ops, Code *code)
{
    bool result = true;
    Nob_String_Builder sb = {0};
    Loop_Patches loops = {0};
    Addrs divide_calls = {0};
    Addrs multiply_calls = {0};
    Addrs compare_calls[5] = {0};
    bool fixed_head = ops.static_positions;
    int64_t static_head = 0;
    Affine_Plan *plans = NULL;
    for (size_t i = 0; i < ops.count; ++i) {
        Op op = ops.items[i];
        int32_t base = (int32_t)static_head;
        {
            size_t end;
            Increment_Carry increment;
            if (increment_carry(ops, i, &end, &increment)) {
                emit_increment_carry(&sb, increment, base);
                static_head += increment.flag;
                if (clear_at(ops, end + 1)) end += 3;
                i = end;
                continue;
            }
        }
        if (op.kind == OP_JUMP_IF_ZERO) {
            size_t close;
            int32_t stride;
            if (scan_loop(ops, i, &close, &stride)) {
                emit_head_adjust(&sb, base);
                static_head = 0;
                emit_scan(&sb, stride);
                i = close;
                continue;
            }
        }
        {
            size_t end;
            if (fixed_integer_block(ops, i, &end)) {
                emit_fixed_integer_call(&sb, base);
                static_head -= 5;
                i = end;
                continue;
            }
            if (fixed_divide_block(ops, i, &end)) {
                emit_fixed_divide_call(&sb, base, &divide_calls);
                static_head -= 3;
                i = end;
                continue;
            }
            Fixed_Add add;
            if (fixed_add_block(ops, i, &end, &add)) {
                emit_fixed_add_call(&sb, add, base);
                static_head += add.final_head;
                i = end;
                continue;
            }
            Fixed_Compare compare;
            if (fixed_compare_block(ops, i, &end, &compare)) {
                emit_fixed_compare_call(&sb, compare, base, compare_calls);
                static_head += compare.final_head;
                i = end;
                continue;
            }
            if (fixed_sqrt_block(ops, i, &end)) {
                emit_fixed_sqrt_call(&sb, base);
                static_head -= 12;
                i = end;
                continue;
            }
            if (fixed_multiply_block(ops, i, &end)) {
                emit_fixed_multiply_call(&sb, base, &multiply_calls);
                static_head -= 7;
                i = end;
                continue;
            }
        }
        if (op.kind == OP_JUMP_IF_ZERO) {
            size_t end;
            Modulo_256 modulo;
            if (modulo_256_loop(ops, i, &end, &modulo)) {
                emit_modulo_256(&sb, modulo, base);
                i = end;
                continue;
            }
            Subdigit subdigit;
            if (subdigit_block(ops, i, &end, &subdigit)) {
                emit_subdigit(&sb, subdigit, base);
                static_head += 2;
                i = end;
                continue;
            }
            if (division_digit_loop(ops, i, &end)) {
                emit_division_digit(&sb, base);
                i = end;
                continue;
            }
            Multi_Multiply multiply;
            if (multi_multiply_loop(ops, i, &end, &multiply)) {
                emit_multi_multiply(&sb, multiply, base);
                i = end;
                continue;
            }
            Byte_Compare compare;
            if (byte_compare_loop(ops, i, &end, &compare)) {
                emit_byte_compare(&sb, compare, base);
                i = end;
                continue;
            }
            Multi_Add add;
            if (multi_add_loop(ops, i, &end, &add)) {
                emit_multi_add(&sb, add, base);
                i = end;
                continue;
            }
        }
        if (op.kind == OP_JUMP_IF_ZERO) {
            size_t end;
            Truth truth;
            if (truth_block(ops, i, &end, &truth)) {
                emit_truth(&sb, truth, base);
                static_head += truth.source;
                i = end;
                continue;
            }
            Complement complement;
            if (complement_block(ops, i, &end, &complement)) {
                emit_complement(&sb, complement, base);
                i = end;
                continue;
            }
        }
        if (op.kind == OP_JUMP_IF_ZERO) {
            size_t end;
            Copy copy;
            if (copy_block(ops, i, &end, &copy) || move_block(ops, i, &end, &copy)) {
                emit_copy(&sb, copy, base);
                static_head += copy.final_head;
                if (clear_at(ops, end + 1)) end += 3;
                i = end;
                continue;
            }
        }
        if (op.kind == OP_JUMP_IF_ZERO && clear_at(ops, i) &&
            i + 3 < ops.count &&
            (ops.items[i + 3].kind == OP_INC || ops.items[i + 3].kind == OP_DEC)) {
            nob_da_append(&sb, 0xc6);
            emit_modrm_rdi(&sb, 0, base);
            nob_da_append(&sb, (uint8_t)add_amount(ops.items[i + 3]));
            i += 3;
            continue;
        }
        if (op.kind == OP_JUMP_IF_ZERO && i + 2 < ops.count &&
            (ops.items[i + 1].kind == OP_INC || ops.items[i + 1].kind == OP_DEC) &&
            (ops.items[i + 1].operand & 1) &&
            ops.items[i + 2].kind == OP_JUMP_IF_NONZERO &&
            op.operand == i + 3 && ops.items[i + 2].operand == i + 1) {
            emit_zero(&sb, base);
            i += 2;
            continue;
        }
        if (op.kind == OP_JUMP_IF_ZERO) {
            size_t close;
            uint8_t source;
            Deltas deltas = {0};
            if (linear_loop(ops, i, &close, &deltas, &source)) {
                emit_linear(&sb, deltas, source, base);
                nob_da_free(deltas);
                if (clear_at(ops, close + 1)) close += 3;
                i = close;
                continue;
            }
        }
        if (op.kind == OP_JUMP_IF_ZERO) {
            size_t close;
            Carry carry;
            if (carry_loop(ops, i, &close, &carry)) {
                emit_carry(&sb, carry, base);
                if (clear_at(ops, close + 1)) close += 3;
                i = close;
                continue;
            }
        }
        if (op.kind == OP_JUMP_IF_ZERO) {
            size_t close;
            Affine *affine = NULL;
            if (affine_loop(ops, i, &close, &affine, &plans)) {
                if (emit_small_affine_if(&sb, affine, base) ||
                    emit_simple_affine(&sb, affine, base)) free(affine);
                else emit_affine_call(&sb, affine, base);
                if (clear_at(ops, close + 1)) close += 3;
                i = close;
                continue;
            }
        }
        switch (op.kind) {
            case OP_INC: {
                assert(op.operand < 256 && "TODO: support bigger operands");
                nob_da_append(&sb, 0x80); // add byte [rdi+base],
                emit_modrm_rdi(&sb, 0, base);
                nob_da_append(&sb, op.operand&0xFF);
            } break;

            case OP_DEC: {
                assert(op.operand < 256 && "TODO: support bigger operands");
                nob_da_append(&sb, 0x80); // sub byte [rdi+base],
                emit_modrm_rdi(&sb, 5, base);
                nob_da_append(&sb, op.operand&0xFF);
            } break;

            // TODO: range checks for OP_LEFT and OP_RIGHT
            case OP_LEFT: {
                static_head -= op.operand;
            } break;

            case OP_RIGHT: {
                static_head += op.operand;
            } break;

            case OP_OUTPUT: {
                if (fixed_head && op.operand >= 8) {
                    nob_da_append(&sb, 0x57);                // push rdi
                    nob_da_append_many(&sb, "\x0f\xb6", 2); // movzx edi, byte [cell]
                    emit_modrm_rdi(&sb, 7, base);
                    nob_da_append(&sb, 0xbe);                // mov esi, count
                    nob_da_append_many(&sb, &op.operand, sizeof(op.operand));
                    void (*helper)(uint8_t, uint32_t) = output_repeat;
                    emit_call(&sb, fnaddr(&helper, sizeof(helper)));
                    nob_da_append(&sb, 0x5f);                // pop rdi
                    break;
                }
                for (size_t i = 0; i < op.operand; ++i) {
                    nob_da_append(&sb, 0x57);                    // push rdi
                    nob_da_append_many(&sb, "\x0f\xb6", 2);     // movzx edi, byte [cell]
                    emit_modrm_rdi(&sb, 7, base);
                    int (*helper)(int) = fixed_head ? putchar_unlocked : putchar;
                    emit_call(&sb, fnaddr(&helper, sizeof(helper)));
                    nob_da_append(&sb, 0x5f);                    // pop rdi
                }
            } break;

            case OP_INPUT: {
                if (op.operand > 1) {
                    nob_da_append(&sb, 0x57); // push rdi
                    if (base) {
                        nob_sb_append_cstr(&sb, "\x48\x8d\xbf"); // lea rdi, [rdi+base]
                        nob_da_append_many(&sb, &base, sizeof(base));
                    }
                    nob_da_append(&sb, 0xbe); // mov esi, count
                    nob_da_append_many(&sb, &op.operand, sizeof(op.operand));
                    void (*helper)(uint8_t *, uint32_t) = fixed_head ?
                        input_repeat : input_repeat_locked;
                    emit_call(&sb, fnaddr(&helper, sizeof(helper)));
                    nob_da_append(&sb, 0x5f); // pop rdi
                    break;
                }
                for (size_t i = 0; i < op.operand; ++i) {
                    nob_da_append(&sb, 0x57);                // push rdi
                    int (*helper)(void) = fixed_head ? getchar_unlocked : getchar;
                    emit_call(&sb, fnaddr(&helper, sizeof(helper)));
                    nob_da_append(&sb, 0x5f);                // pop rdi
                    nob_da_append_many(&sb, "\x83\xf8\xff\x74\x00", 5);
                    size_t skip = sb.count - 1;
                    emit_store_byte(&sb, 0, base);
                    sb.items[skip] = (uint8_t)(sb.count - skip - 1);
                }
            } break;

            case OP_JUMP_IF_ZERO: {
                if (!fixed_head) {
                    emit_head_adjust(&sb, base);
                    static_head = 0;
                    base = 0;
                }
                nob_da_append(&sb, 0x80);                  // cmp byte [rdi+base], 0
                emit_modrm_rdi(&sb, 7, base);
                nob_da_append(&sb, 0);
                nob_sb_append_cstr(&sb, "\x0f\x84");     // jz
                size_t operand_byte_addr = sb.count;
                nob_da_append_many(&sb, "\x00\x00\x00\x00", 4);
                Loop_Patch loop = {
                    .operand_byte_addr = operand_byte_addr,
                    .body_byte_addr = sb.count,
                };
                nob_da_append(&loops, loop);
            } break;

            case OP_JUMP_IF_NONZERO: {
                if (!fixed_head) {
                    emit_head_adjust(&sb, base);
                    static_head = 0;
                    base = 0;
                }
                nob_da_append(&sb, 0x80);                  // cmp byte [rdi+base], 0
                emit_modrm_rdi(&sb, 7, base);
                nob_da_append(&sb, 0);
                nob_sb_append_cstr(&sb, "\x0f\x85");     // jnz
                size_t operand_byte_addr = sb.count;
                nob_da_append_many(&sb, "\x00\x00\x00\x00", 4);
                size_t src_byte_addr = sb.count;
                assert(loops.count > 0);
                Loop_Patch loop = loops.items[--loops.count];
                int32_t relative = (int32_t)(loop.body_byte_addr - src_byte_addr);
                memcpy(&sb.items[operand_byte_addr], &relative, sizeof(relative));
                relative = (int32_t)(sb.count - loop.body_byte_addr);
                memcpy(&sb.items[loop.operand_byte_addr], &relative, sizeof(relative));
            } break;

            case OP_FORK: {
                if (!fixed_head) {
                    emit_head_adjust(&sb, base);
                    static_head = 0;
                }
                nob_sb_append_cstr(&sb, "\x57");                    // save parent head
                nob_da_append_many(&sb, "\x48\x8d\x7f\x01", 4);    // child head = parent + 1
                nob_sb_append_cstr(&sb, "\x48\x8d\x35");            // child entry
                size_t patch = sb.count;
                uint32_t zero = 0;
                nob_da_append_many(&sb, &zero, sizeof(zero));
                size_t source = sb.count;
                nob_sb_append_cstr(&sb, "\xba");                    // site
                uint32_t site = op.operand;
                nob_da_append_many(&sb, &site, sizeof(site));
                void (*helper)(uint8_t *, Jit_Fn, uint32_t) = fork_thread;
                emit_call(&sb, fnaddr(&helper, sizeof(helper)));
                nob_sb_append_cstr(&sb, "\x5f");                    // restore parent head
                int32_t relative = (int32_t)(sb.count - source);
                memcpy(&sb.items[patch], &relative, sizeof(relative));
            } break;

            case OP_CHILD_END: {
                if (!fixed_head) {
                    emit_head_adjust(&sb, base);
                    static_head = 0;
                }
                nob_sb_append_cstr(&sb, "\x57\xbf");                // save head; site
                uint32_t site = op.operand;
                nob_da_append_many(&sb, &site, sizeof(site));
                int (*helper)(uint32_t) = child_ends_here;
                emit_call(&sb, fnaddr(&helper, sizeof(helper)));
                nob_da_append_many(&sb, "\x85\xc0\x5f\x74\x01\xc3", 6);
            } break;

            case OP_JOIN: {
                if (!fixed_head) {
                    emit_head_adjust(&sb, base);
                    static_head = 0;
                }
                nob_sb_append_cstr(&sb, "\x57\xbf");                // save head; site
                uint32_t site = op.operand;
                nob_da_append_many(&sb, &site, sizeof(site));
                void (*helper)(uint32_t) = join_thread;
                emit_call(&sb, fnaddr(&helper, sizeof(helper)));
                nob_sb_append_cstr(&sb, "\x5f");                    // restore head
            } break;

            default: assert(0 && "Unreachable");
        }
    }
    assert(loops.count == 0);

    nob_sb_append_cstr(&sb, "\xC3");

    if (divide_calls.count) {
        size_t divide_stub = sb.count;
        static const uint8_t divide_code[] = {
            0x8b,0x46,0x04,0x8b,0x4e,0x08,0x41,0x89,0xc0,0x41,0x89,0xc9,
            0x41,0x89,0xc2,0x41,0xf7,0xda,0x44,0x0f,0x48,0xd0,0x41,0x89,
            0xcb,0x41,0xf7,0xdb,0x44,0x0f,0x48,0xd9,0x45,0x85,0xdb,0x74,
            0x20,0x44,0x89,0xd0,0x89,0xc2,0xc1,0xea,0x10,0xc1,0xe0,0x10,
            0x44,0x39,0xda,0x72,0x0b,0x89,0xc1,0x89,0xd0,0x31,0xd2,0x41,
            0xf7,0xf3,0x89,0xc8,0x41,0xf7,0xf3,0xeb,0x11,0x44,0x89,0xd2,
            0x48,0xc1,0xe2,0x10,0x48,0xc1,0xe2,0x18,0x48,0xc1,0xea,0x18,
            0x31,0xc0,0x89,0xc1,0xf7,0xd9,0x45,0x31,0xc8,0x0f,0x48,0xc1,
            0x89,0x06,0x44,0x89,0x56,0x04,0x44,0x89,0x5e,0x08,0x89,0x56,
            0x0c,0x48,0xc1,0xea,0x20,0x88,0x56,0x10,0x31,0xc9,0x48,0x89,
            0x4e,0x11,0x48,0x89,0x4e,0x14,0xc3,
        };
        nob_da_append_many(&sb, divide_code, sizeof(divide_code));
        for (size_t i = 0; i < divide_calls.count; ++i) {
            size_t operand = divide_calls.items[i];
            int32_t relative = (int32_t)(divide_stub - (operand + 4));
            memcpy(sb.items + operand, &relative, sizeof(relative));
        }
    }

    if (multiply_calls.count) {
        size_t multiply_stub = sb.count;
        static const uint8_t multiply_code[] = {
            0x8b,0x46,0x04,                         // mov eax, [rsi+4]
            0x44,0x8b,0x4e,0x08,                    // mov r9d, [rsi+8]
            0x48,0xc7,0x46,0x14,0,0,0,0,            // mov qword [rsi+20], 0
            0x89,0xc2,                               // mov edx, eax
            0x44,0x89,0xc9,                          // mov ecx, r9d
            0xf7,0xda,                               // neg edx
            0x0f,0x48,0xd0,                          // cmovs edx, eax
            0xf7,0xd9,                               // neg ecx
            0x41,0x0f,0x48,0xc9,                     // cmovs ecx, r9d
            0x89,0x56,0x04,                          // mov [rsi+4], edx
            0x89,0x4e,0x08,                          // mov [rsi+8], ecx
            0x48,0x0f,0xaf,0xd1,                     // imul rdx, rcx
            0x48,0x89,0x56,0x0c,                     // mov [rsi+12], rdx
            0x48,0xc1,0xea,0x10,                     // shr rdx, 16
            0x41,0x89,0xd0,                          // mov r8d, edx
            0xf7,0xda,                               // neg edx
            0x44,0x31,0xc8,                          // xor eax, r9d
            0x44,0x0f,0x48,0xc2,                     // cmovs r8d, edx
            0x44,0x89,0x06,                          // mov [rsi], r8d
            0xc3,                                    // ret
        };
        nob_da_append_many(&sb, multiply_code, sizeof(multiply_code));
        for (size_t i = 0; i < multiply_calls.count; ++i) {
            size_t operand = multiply_calls.items[i];
            int32_t relative = (int32_t)(multiply_stub - (operand + 4));
            memcpy(sb.items + operand, &relative, sizeof(relative));
        }
    }

    static const uint8_t compare_conditions[] = {0x9c, 0x9e, 0x9f, 0x9d, 0x94};
    for (size_t comparison = 0;
         comparison < NOB_ARRAY_LEN(compare_calls); ++comparison) {
        if (!compare_calls[comparison].count) continue;
        size_t compare_stub = sb.count;
        static const uint8_t before_condition[] = {
            0x31,0xc0,                               // xor eax, eax
            0x48,0x89,0x46,0x0c,                    // mov [rsi+12], rax
            0x48,0x89,0x46,0x12,                    // mov [rsi+18], rax
            0x89,0x06,                               // mov [rsi], eax
            0x8b,0x4e,0x08,                          // mov ecx, [rsi+8]
            0x39,0x4e,0x04,                          // cmp [rsi+4], ecx
            0x48,0xb8,0,0,0,0x80,0,0,0,0x80,        // movabs rax, sign bits
            0x0f,
        };
        static const uint8_t after_condition[] = {
            0x06,                                    // setcc byte [rsi]
            0x48,0x31,0x46,0x04,                    // xor [rsi+4], rax
            0xc3,                                    // ret
        };
        nob_da_append_many(&sb, before_condition, sizeof(before_condition));
        nob_da_append(&sb, compare_conditions[comparison]);
        nob_da_append_many(&sb, after_condition, sizeof(after_condition));
        for (size_t i = 0; i < compare_calls[comparison].count; ++i) {
            size_t operand = compare_calls[comparison].items[i];
            int32_t relative = (int32_t)(compare_stub - (operand + 4));
            memcpy(sb.items + operand, &relative, sizeof(relative));
        }
    }

    code->len = sb.count;
    code->run = mmap(NULL, sb.count, PROT_EXEC | PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (code->run == MAP_FAILED) {
        nob_log(NOB_ERROR, "Could not allocate executable memory: %s", strerror(errno));
        nob_return_defer(false);
    }

    // TODO: switch the permissions to only-exec after finishing copying the code. See mprotect(2).
    memcpy(code->run, sb.items, code->len);

defer:
    if (!result) {
        free_code(*code);
        memset(code, 0, sizeof(*code));
    }
    nob_da_free(sb);
    nob_da_free(loops);
    nob_da_free(divide_calls);
    nob_da_free(multiply_calls);
    for (size_t i = 0; i < NOB_ARRAY_LEN(compare_calls); ++i)
        nob_da_free(compare_calls[i]);
    return result;
}

bool generate_ops(const char *file_path, Ops *ops)
{
    bool result = true;
    Addrs stack = {0};
    Forks forks = {0};
    uint32_t next_site = 0;
    int64_t static_head = 0;
    ops->static_positions = true;
    int fd = open(file_path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        nob_log(NOB_ERROR, "Could not open %s: %s", file_path, strerror(errno));
        return false;
    }
    struct stat statbuf;
    if (fstat(fd, &statbuf) < 0 || statbuf.st_size < 0) {
        close(fd);
        nob_log(NOB_ERROR, "Could not stat %s: %s", file_path, strerror(errno));
        return false;
    }
    size_t source_count = (size_t)statbuf.st_size;
    char *source = NULL;
    if (source_count) {
        source = mmap(NULL, source_count, PROT_READ, MAP_PRIVATE, fd, 0);
        if (source == MAP_FAILED) {
            close(fd);
            nob_log(NOB_ERROR, "Could not map %s: %s", file_path, strerror(errno));
            return false;
        }
        madvise(source, source_count, MADV_SEQUENTIAL);
        if (!ops->items && source_count / 8 < SIZE_MAX / sizeof(*ops->items)) {
            size_t capacity = source_count / 8 + 1;
            Op *items = malloc(capacity * sizeof(*items));
            if (items) {
                ops->items = items;
                ops->capacity = capacity;
            }
        }
    }
    close(fd);
    Lexer l = {
        .content = {
            .data = source,
            .count = source_count,
        },
    };
    char c = lexer_next(&l);
    while (c) {
        switch (c) {
            case '.':
            case ',':
            case '<':
            case '>':
            case '-':
            case '+': {
                size_t start = l.pos;
                uint64_t repeated = UINT64_C(0x0101010101010101) * (uint8_t)c;
                while (l.content.count - l.pos >= sizeof(repeated)) {
                    uint64_t word;
                    memcpy(&word, l.content.data + l.pos, sizeof(word));
                    uint64_t difference = word ^ repeated;
                    if (difference) {
                        l.pos += (size_t)__builtin_ctzll(difference) / 8;
                        goto run_done;
                    }
                    l.pos += sizeof(word);
                }
                while (l.pos < l.content.count && l.content.data[l.pos] == c)
                    l.pos += 1;
run_done:
                ;
                size_t count = l.pos - start + 1;
                char s = lexer_next(&l);
                if (count > INT32_MAX) {
                    printf("%s [%zu]: ERROR: Command run is too long\n", file_path, l.pos);
                    nob_return_defer(false);
                }
                Op op = {
                    .kind = c,
                    .operand = (uint32_t)count,
                };
                nob_da_append(ops, op);
                if (op.kind == OP_LEFT) static_head -= op.operand;
                else if (op.kind == OP_RIGHT) static_head += op.operand;
                if (static_head < INT32_MIN || static_head > INT32_MAX)
                    ops->static_positions = false;
                c = s;
            } break;

            case '[': {
                size_t addr = ops->count;
                Op op = {
                    .kind = c,
                    .operand = ops->static_positions ?
                        (uint32_t)(int32_t)static_head : 0,
                };
                nob_da_append(ops, op);
                nob_da_append(&stack, addr);

                c = lexer_next(&l);
            } break;

            case ']': {
                if (stack.count == 0) {
                    // TODO: reports rows and columns
                    printf("%s [%zu]: ERROR: Unbalanced loop\n", file_path, l.pos);
                    nob_return_defer(false);
                }

                size_t addr = stack.items[--stack.count];
                if (ops->static_positions &&
                    (int32_t)ops->items[addr].operand != static_head)
                    ops->static_positions = false;
                Op op = {
                    .kind = c,
                    .operand = (uint32_t)(addr + 1),
                };
                nob_da_append(ops, op);
                if (ops->count > INT32_MAX) {
                    printf("%s [%zu]: ERROR: Program has too many operations\n",
                           file_path, l.pos);
                    nob_return_defer(false);
                }
                ops->items[addr].operand = (uint32_t)ops->count;

                c = lexer_next(&l);
            } break;

            case '(': {
                uint32_t site = next_site++;
                ops->static_positions = false;
                nob_da_append(ops, ((Op){.kind = OP_FORK, .operand = site}));
                nob_da_append(&forks, ((Fork){.site = site}));
                c = lexer_next(&l);
            } break;

            case ')': {
                if (!forks.count || forks.items[forks.count - 1].closed) {
                    c = lexer_next(&l); // ordinary BF comment punctuation
                    break;
                }
                Fork *fork = &forks.items[forks.count - 1];
                fork->closed = true;
                nob_da_append(ops, ((Op){.kind = OP_CHILD_END, .operand = fork->site}));
                c = lexer_next(&l);
            } break;

            case '#': {
                if (!forks.count) {
                    c = lexer_next(&l); // ordinary BF comment punctuation
                    break;
                }
                Fork fork = forks.items[forks.count - 1];
                if (!fork.closed) {
                    printf("%s [%zu]: ERROR: Join before child end\n", file_path, l.pos);
                    nob_return_defer(false);
                }
                forks.count -= 1;
                nob_da_append(ops, ((Op){.kind = OP_JOIN, .operand = fork.site}));
                c = lexer_next(&l);
            } break;

            default: {}
        }
    }

    if (stack.count > 0) {
        // TODO: report the location of opening unbalanced bracket
        printf("%s [%zu]: ERROR: Unbalanced loop\n", file_path, l.pos);
        nob_return_defer(false);
    }
    for (size_t i = 0; i < forks.count; ++i) {
        if (!forks.items[i].closed) {
            printf("%s [%zu]: ERROR: Unbalanced fork\n", file_path, l.pos);
            nob_return_defer(false);
        }
    }

defer:
    if (!result) {
        nob_da_free(*ops);
        memset(ops, 0, sizeof(*ops));
    }
    if (source_count) munmap(source, source_count);
    nob_da_free(stack);
    nob_da_free(forks);
    return result;
}

void usage(const char *program)
{
    nob_log(NOB_ERROR, "Usage: %s [--no-jit] <input.bf>", program);
}

int main(int argc, char **argv)
{
    int result = 0;
    Ops ops = {0};
    Code code = {0};
    void *memory = NULL;

    const char *program = nob_shift_args(&argc, &argv);

    bool no_jit = false;
    const char *file_path = NULL;

    while (argc > 0) {
        const char *flag = nob_shift_args(&argc, &argv);
        if (strcmp(flag, "--no-jit") == 0) {
            no_jit = true;
        } else {
            if (file_path != NULL) {
                usage(program);
                // TODO(multifile): what if we allowed providing several files and executed them sequencially
                // preserving the state of the machine between them? Maybe complicated by TODO(dead).
                nob_log(NOB_ERROR, "Providing several files is not supported");
                nob_return_defer(1);
            }

            file_path = flag;
        }
    }

    if (file_path == NULL) {
        usage(program);
        nob_log(NOB_ERROR, "No input is provided");
        nob_return_defer(1);
    }

    if (!generate_ops(file_path, &ops)) nob_return_defer(1);

    if (no_jit) {
        nob_log(NOB_INFO, "JIT: off");
        if (!interpret(ops)) nob_return_defer(1);
    } else {
        nob_log(NOB_INFO, "JIT: on");
        if (!jit_compile(ops, &code)) nob_return_defer(1);
        memory = malloc(JIT_MEMORY_CAP);
        memset(memory, 0, JIT_MEMORY_CAP);
        assert(memory != NULL);
        Thread root = {.stop_at = UINT32_MAX};
        thread = &root;
        code.run((uint8_t *)memory + JIT_MEMORY_CAP / 2);
        while (root.joins) join_thread(root.joins->site);
    }

defer:
    nob_da_free(ops);
    free_code(code);
    free(memory);
    return result;
}

// TODO: Add more interesting examples.
//   Check https://brainfuck.org/ for inspiration
// TODO(dead): Dead code eliminate first loop which traditionally used as a comment.
//   May not work well if we start sequencially executing several files,
//   because consequent files may not start from the zero state.
//   See TODO(multifile).
// TODO: Optimize pattern [-] to just set the current cell to 0.
//   Probably on the level of IR.
// TODO: Windows port.
//   - [ ] Platform specific mapping of executable memory
//   - [ ] Platform specific stdio from JIT compiled machine code
