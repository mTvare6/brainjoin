#include <stdio.h>
#include <stdint.h>
#include <pthread.h>

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
    uint8_t tables[];
} Affine_Plan;

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
    for (size_t row = 0; row < width; ++row) {
        for (size_t column = 0; column < width; ++column) {
            uint32_t sum = 0;
            for (size_t k = 0; k < width; ++k)
                sum += (uint32_t)left[row * width + k] * right[k * width + column];
            out[row * width + column] = (uint8_t)sum;
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
    Affine_Plan *plan = malloc(sizeof(*plan) + table_count * cells);
    if (!plan) abort();
    plan->next = *plans;
    plan->count = count;
    plan->kind = kind;
    if (kind == AFFINE_IF) {
        memcpy(plan->tables, matrix, cells);
    } else {
        memset(plan->tables, 0, cells);
        for (size_t i = 0; i < width; ++i) plan->tables[i * width + i] = 1;
        memcpy(plan->tables + cells, matrix, cells);
        for (size_t i = 2; i < 256; ++i)
            matrix_multiply(plan->tables + i * cells, matrix,
                            plan->tables + (i - 1) * cells, width);
    }
    *plans = plan;
    return plan;
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
        A(1,-1) && to_low < 0 && A(3,1) && low_to_scratch > 0 &&
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
    uint8_t state[33], next[33];
    for (size_t i = 0; i < count; ++i) state[i] = base[affine->offsets[i]];
    state[count] = 1;
    uint8_t iterations = state[affine->control];
    if (!iterations) return;
    if (affine->kind == AFFINE_UP) iterations = (uint8_t)-iterations;
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

static bool static_positions(Ops ops, int32_t **result)
{
    int32_t *positions = malloc(ops.count * sizeof(*positions));
    int64_t *loops = NULL;
    size_t loop_count = 0, loop_capacity = 0;
    if (ops.count && !positions) abort();
    int64_t head = 0;
    for (size_t i = 0; i < ops.count; ++i) {
        if (head < INT32_MIN || head > INT32_MAX) goto fail;
        positions[i] = (int32_t)head;
        Op op = ops.items[i];
        if (op.kind == OP_LEFT) head -= op.operand;
        else if (op.kind == OP_RIGHT) head += op.operand;
        else if (op.kind == OP_JUMP_IF_ZERO) {
            if (loop_count >= loop_capacity) {
                loop_capacity = loop_capacity ? loop_capacity * 2 : 256;
                loops = realloc(loops, loop_capacity * sizeof(*loops));
                if (!loops) abort();
            }
            loops[loop_count++] = head;
        } else if (op.kind == OP_JUMP_IF_NONZERO) {
            if (!loop_count || loops[--loop_count] != head) goto fail;
        } else if (op.kind == OP_FORK || op.kind == OP_CHILD_END || op.kind == OP_JOIN) {
            goto fail;
        }
    }
    if (loop_count || head < INT32_MIN || head > INT32_MAX) goto fail;
    free(loops);
    *result = positions;
    return true;

fail:
    free(loops);
    free(positions);
    return false;
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

static void emit_affine_call(Nob_String_Builder *sb, const Affine *affine,
                             int32_t base)
{
    nob_da_append(sb, 0x57); // push rdi
    if (base) {
        nob_sb_append_cstr(sb, "\x48\x8d\xbf"); // lea rdi, [rdi+base]
        nob_da_append_many(sb, &base, sizeof(base));
    }
    nob_sb_append_cstr(sb, "\x48\xbe"); // movabs rsi, affine
    uintptr_t descriptor = (uintptr_t)affine;
    nob_da_append_many(sb, &descriptor, sizeof(descriptor));
    void (*helper)(uint8_t *, const Affine *) = affine_exec;
    emit_call(sb, fnaddr(&helper, sizeof(helper)));
    nob_da_append(sb, 0x5f); // pop rdi
}

bool jit_compile(Ops ops, Code *code)
{
    bool result = true;
    Nob_String_Builder sb = {0};
    Loop_Patches loops = {0};
    int32_t *positions = NULL;
    bool fixed_head = static_positions(ops, &positions);
    Affine_Plan *plans = NULL;

    for (size_t i = 0; i < ops.count; ++i) {
        Op op = ops.items[i];
        int32_t base = fixed_head ? positions[i] : 0;
        {
            size_t end;
            Increment_Carry increment;
            if (increment_carry(ops, i, &end, &increment)) {
                emit_increment_carry(&sb, increment, base);
                if (!fixed_head) emit_head_adjust(&sb, increment.flag);
                if (clear_at(ops, end + 1)) end += 3;
                i = end;
                continue;
            }
        }
        if (op.kind == OP_JUMP_IF_ZERO) {
            size_t end;
            Truth truth;
            if (truth_block(ops, i, &end, &truth)) {
                emit_truth(&sb, truth, base);
                if (!fixed_head) emit_head_adjust(&sb, truth.source);
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
                if (!fixed_head) emit_head_adjust(&sb, copy.final_head);
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
                if (emit_simple_affine(&sb, affine, base)) free(affine);
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
                if (!fixed_head) {
                    nob_sb_append_cstr(&sb, "\x48\x81\xef"); // sub rdi,
                    uint32_t operand = (uint32_t)op.operand;
                    nob_da_append_many(&sb, &operand, sizeof(operand));
                }
            } break;

            case OP_RIGHT: {
                if (!fixed_head) {
                    nob_sb_append_cstr(&sb, "\x48\x81\xc7"); // add rdi,
                    uint32_t operand = (uint32_t)op.operand;
                    nob_da_append_many(&sb, &operand, sizeof(operand));
                }
            } break;

            case OP_OUTPUT: {
                for (size_t i = 0; i < op.operand; ++i) {
                    nob_sb_append_cstr(&sb, "\x57");                            // push rdi
                    nob_da_append_many(&sb, "\x48\xc7\xc0\x01\x00\x00\x00", 7); // mov rax, 1
                    if (fixed_head) {
                        nob_sb_append_cstr(&sb, "\x48\x8d\xb7");                // lea rsi, [rdi+base]
                        nob_da_append_many(&sb, &base, sizeof(base));
                    } else nob_sb_append_cstr(&sb, "\x48\x89\xfe");            // mov rsi, rdi
                    nob_da_append_many(&sb, "\x48\xc7\xc7\x01\x00\x00\x00", 7); // mov rdi, 1
                    nob_da_append_many(&sb, "\x48\xc7\xc2\x01\x00\x00\x00", 7); // mov rdx, 1
                    nob_sb_append_cstr(&sb, "\x0f\x05");                        // syscall
                    nob_sb_append_cstr(&sb, "\x5f");                            // pop rdi
                }
            } break;

            case OP_INPUT: {
                for (size_t i = 0; i < op.operand; ++i) {
                    nob_sb_append_cstr(&sb, "\x57");                            // push rdi
                    nob_da_append_many(&sb, "\x48\xc7\xc0\x00\x00\x00\x00", 7); // mov rax, 0
                    if (fixed_head) {
                        nob_sb_append_cstr(&sb, "\x48\x8d\xb7");                // lea rsi, [rdi+base]
                        nob_da_append_many(&sb, &base, sizeof(base));
                    } else nob_sb_append_cstr(&sb, "\x48\x89\xfe");            // mov rsi, rdi
                    nob_da_append_many(&sb, "\x48\xc7\xc7\x00\x00\x00\x00", 7); // mov rdi, 0
                    nob_da_append_many(&sb, "\x48\xc7\xc2\x01\x00\x00\x00", 7); // mov rdx, 1
                    nob_sb_append_cstr(&sb, "\x0f\x05");                        // syscall
                    nob_sb_append_cstr(&sb, "\x5f");                            // pop rdi
                }
            } break;

            case OP_JUMP_IF_ZERO: {
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
                nob_sb_append_cstr(&sb, "\x57\xbf");                // save head; site
                uint32_t site = op.operand;
                nob_da_append_many(&sb, &site, sizeof(site));
                int (*helper)(uint32_t) = child_ends_here;
                emit_call(&sb, fnaddr(&helper, sizeof(helper)));
                nob_da_append_many(&sb, "\x85\xc0\x5f\x74\x01\xc3", 6);
            } break;

            case OP_JOIN: {
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
    free(positions);
    return result;
}

bool generate_ops(const char *file_path, Ops *ops)
{
    bool result = true;
    Addrs stack = {0};
    Forks forks = {0};
    uint32_t next_site = 0;
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
                while (l.pos < l.content.count && l.content.data[l.pos] == c)
                    l.pos += 1;
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
                c = s;
            } break;

            case '[': {
                size_t addr = ops->count;
                Op op = {
                    .kind = c,
                    .operand = 0,
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
        code.run(memory);
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
