// muguet - mem32.rs
// rewrites game.wasm from a 64-bit to a 32-bit linear memory (ios only).
// a memory64 needs an explicit bounds check on every access; a 32-bit memory with a 4 GiB
// reservation lets wasmtime drop them and rely on guard pages instead. the game's heap is 3 GiB,
// so every address fits in 32 bits: addresses are computed as i64 like before and wrapped to
// i32 right before each memory instruction.

use anyhow::{bail, Result};
use wasm_encoder::reencode::{utils, Error, Reencode};
use wasm_encoder::{Function, Instruction as I, MemArg, ValType};
use wasmparser::{Parser, Payload};

const MAX_PAGES_32: u64 = 65536;

struct Mem32 {
    // parameter count of every function, by function index (imports first)
    func_params: Vec<u32>,
    next_func: u32,
}

// scratch locals appended to every function, relative to the first one
const T_I32A: u32 = 0;
const T_I32B: u32 = 1;
const T_I64A: u32 = 2;
const T_I64B: u32 = 3;
const T_F32: u32 = 4;
const T_F64: u32 = 5;
const T_V128: u32 = 6;
const SCRATCH: [ValType; 7] = [ValType::I32, ValType::I32, ValType::I64, ValType::I64, ValType::F32, ValType::F64, ValType::V128];

impl Reencode for Mem32 {
    type Error = anyhow::Error;

    fn memory_type(&mut self, m: wasmparser::MemoryType) -> Result<wasm_encoder::MemoryType, Error<Self::Error>> {
        let mut t = utils::memory_type(self, m);
        if t.memory64 {
            if t.minimum > MAX_PAGES_32 {
                return Err(Error::UserError(anyhow::anyhow!("memory needs {} pages at start, more than a 32-bit memory holds", t.minimum)));
            }
            t.memory64 = false;
            t.maximum = Some(t.maximum.unwrap_or(MAX_PAGES_32).min(MAX_PAGES_32));
        }
        Ok(t)
    }

    fn parse_data(&mut self, data: &mut wasm_encoder::DataSection, datum: wasmparser::Data<'_>) -> Result<(), Error<Self::Error>> {
        match datum.kind {
            wasmparser::DataKind::Active { memory_index, offset_expr } => {
                let mut ops = offset_expr.get_operators_reader();
                let off = match ops.read() {
                    Ok(wasmparser::Operator::I64Const { value }) if value >= 0 && (value as u64) < (MAX_PAGES_32 << 16) => value as u32,
                    other => return Err(Error::UserError(anyhow::anyhow!("unsupported data segment offset {other:?}"))),
                };
                data.active(memory_index, &wasm_encoder::ConstExpr::i32_const(off as i32), datum.data.iter().copied());
            }
            wasmparser::DataKind::Passive => { data.passive(datum.data.iter().copied()); }
        }
        Ok(())
    }

    fn parse_function_body(&mut self, code: &mut wasm_encoder::CodeSection, func: wasmparser::FunctionBody<'_>) -> Result<(), Error<Self::Error>> {
        let index = self.next_func as usize;
        self.next_func += 1;
        let mut locals: Vec<(u32, ValType)> = Vec::new();
        let mut count = self.func_params[index];
        for l in func.get_locals_reader()? {
            let (n, ty) = l?;
            locals.push((n, self.val_type(ty)?));
            count += n;
        }
        let base = count;
        for ty in SCRATCH { locals.push((1, ty)); }
        let mut f = Function::new(locals);
        let mut reader = func.get_operators_reader()?;
        while !reader.eof() {
            let ins = self.parse_instruction(&mut reader)?;
            emit(&mut f, ins, base).map_err(Error::UserError)?;
        }
        code.function(&f);
        Ok(())
    }
}

fn check(m: &MemArg) -> Result<()> {
    if m.offset > u32::MAX as u64 { bail!("memory offset {} does not fit a 32-bit memory", m.offset); }
    Ok(())
}

// address on top of the stack
fn wrap_top(f: &mut Function, ins: &I) {
    f.instruction(&I::I32WrapI64);
    f.instruction(ins);
}

// address under one value of type `t`
fn wrap_under1(f: &mut Function, ins: &I, t: u32) {
    f.instruction(&I::LocalSet(t));
    f.instruction(&I::I32WrapI64);
    f.instruction(&I::LocalGet(t));
    f.instruction(ins);
}

// address under two values (deeper one saved in `a`, top in `b`)
fn wrap_under2(f: &mut Function, ins: &I, a: u32, b: u32) {
    f.instruction(&I::LocalSet(b));
    f.instruction(&I::LocalSet(a));
    f.instruction(&I::I32WrapI64);
    f.instruction(&I::LocalGet(a));
    f.instruction(&I::LocalGet(b));
    f.instruction(ins);
}

fn emit(f: &mut Function, ins: I, base: u32) -> Result<()> {
    let (i32a, i32b, i64a, i64b, f32t, f64t, v128) = (base + T_I32A, base + T_I32B, base + T_I64A, base + T_I64B, base + T_F32, base + T_F64, base + T_V128);
    match &ins {
        // loads: address on top
        I::I32Load(m) | I::I64Load(m) | I::F32Load(m) | I::F64Load(m)
        | I::I32Load8S(m) | I::I32Load8U(m) | I::I32Load16S(m) | I::I32Load16U(m)
        | I::I64Load8S(m) | I::I64Load8U(m) | I::I64Load16S(m) | I::I64Load16U(m) | I::I64Load32S(m) | I::I64Load32U(m)
        | I::V128Load(m) | I::V128Load8x8S(m) | I::V128Load8x8U(m) | I::V128Load16x4S(m) | I::V128Load16x4U(m)
        | I::V128Load32x2S(m) | I::V128Load32x2U(m) | I::V128Load8Splat(m) | I::V128Load16Splat(m) | I::V128Load32Splat(m)
        | I::V128Load64Splat(m) | I::V128Load32Zero(m) | I::V128Load64Zero(m)
        | I::I32AtomicLoad(m) | I::I64AtomicLoad(m) | I::I32AtomicLoad8U(m) | I::I32AtomicLoad16U(m)
        | I::I64AtomicLoad8U(m) | I::I64AtomicLoad16U(m) | I::I64AtomicLoad32U(m) => { check(m)?; wrap_top(f, &ins); }

        // address under one i32
        I::I32Store(m) | I::I32Store8(m) | I::I32Store16(m) | I::I32AtomicStore(m) | I::I32AtomicStore8(m) | I::I32AtomicStore16(m)
        | I::I32AtomicRmwAdd(m) | I::I32AtomicRmw8AddU(m) | I::I32AtomicRmw16AddU(m)
        | I::I32AtomicRmwSub(m) | I::I32AtomicRmw8SubU(m) | I::I32AtomicRmw16SubU(m)
        | I::I32AtomicRmwAnd(m) | I::I32AtomicRmw8AndU(m) | I::I32AtomicRmw16AndU(m)
        | I::I32AtomicRmwOr(m) | I::I32AtomicRmw8OrU(m) | I::I32AtomicRmw16OrU(m)
        | I::I32AtomicRmwXor(m) | I::I32AtomicRmw8XorU(m) | I::I32AtomicRmw16XorU(m)
        | I::I32AtomicRmwXchg(m) | I::I32AtomicRmw8XchgU(m) | I::I32AtomicRmw16XchgU(m)
        | I::MemoryAtomicNotify(m) => { check(m)?; wrap_under1(f, &ins, i32a); }

        // address under one i64
        I::I64Store(m) | I::I64Store8(m) | I::I64Store16(m) | I::I64Store32(m)
        | I::I64AtomicStore(m) | I::I64AtomicStore8(m) | I::I64AtomicStore16(m) | I::I64AtomicStore32(m)
        | I::I64AtomicRmwAdd(m) | I::I64AtomicRmw8AddU(m) | I::I64AtomicRmw16AddU(m) | I::I64AtomicRmw32AddU(m)
        | I::I64AtomicRmwSub(m) | I::I64AtomicRmw8SubU(m) | I::I64AtomicRmw16SubU(m) | I::I64AtomicRmw32SubU(m)
        | I::I64AtomicRmwAnd(m) | I::I64AtomicRmw8AndU(m) | I::I64AtomicRmw16AndU(m) | I::I64AtomicRmw32AndU(m)
        | I::I64AtomicRmwOr(m) | I::I64AtomicRmw8OrU(m) | I::I64AtomicRmw16OrU(m) | I::I64AtomicRmw32OrU(m)
        | I::I64AtomicRmwXor(m) | I::I64AtomicRmw8XorU(m) | I::I64AtomicRmw16XorU(m) | I::I64AtomicRmw32XorU(m)
        | I::I64AtomicRmwXchg(m) | I::I64AtomicRmw8XchgU(m) | I::I64AtomicRmw16XchgU(m) | I::I64AtomicRmw32XchgU(m) => { check(m)?; wrap_under1(f, &ins, i64a); }

        I::F32Store(m) => { check(m)?; wrap_under1(f, &ins, f32t); }
        I::F64Store(m) => { check(m)?; wrap_under1(f, &ins, f64t); }
        I::V128Store(m) => { check(m)?; wrap_under1(f, &ins, v128); }
        I::V128Load8Lane { memarg: m, .. } | I::V128Load16Lane { memarg: m, .. } | I::V128Load32Lane { memarg: m, .. } | I::V128Load64Lane { memarg: m, .. }
        | I::V128Store8Lane { memarg: m, .. } | I::V128Store16Lane { memarg: m, .. } | I::V128Store32Lane { memarg: m, .. } | I::V128Store64Lane { memarg: m, .. } => { check(m)?; wrap_under1(f, &ins, v128); }

        // address under two values
        I::I32AtomicRmwCmpxchg(m) | I::I32AtomicRmw8CmpxchgU(m) | I::I32AtomicRmw16CmpxchgU(m) => { check(m)?; wrap_under2(f, &ins, i32a, i32b); }
        I::I64AtomicRmwCmpxchg(m) | I::I64AtomicRmw8CmpxchgU(m) | I::I64AtomicRmw16CmpxchgU(m) | I::I64AtomicRmw32CmpxchgU(m) => { check(m)?; wrap_under2(f, &ins, i64a, i64b); }
        I::MemoryAtomicWait32(m) => { check(m)?; wrap_under2(f, &ins, i32a, i64b); }
        I::MemoryAtomicWait64(m) => { check(m)?; wrap_under2(f, &ins, i64a, i64b); }

        // size and grow count pages; give the game back i64 like it expects
        I::MemorySize(_) => { f.instruction(&ins); f.instruction(&I::I64ExtendI32U); }
        I::MemoryGrow(_) => { f.instruction(&I::I32WrapI64); f.instruction(&ins); f.instruction(&I::I64ExtendI32S); }

        // fill(dst: i64, value: i32, len: i64)
        I::MemoryFill(_) => {
            f.instruction(&I::I32WrapI64).instruction(&I::LocalSet(i32b));
            f.instruction(&I::LocalSet(i32a));
            f.instruction(&I::I32WrapI64).instruction(&I::LocalGet(i32a)).instruction(&I::LocalGet(i32b)).instruction(&ins);
        }
        // copy(dst: i64, src: i64, len: i64)
        I::MemoryCopy { .. } => {
            f.instruction(&I::I32WrapI64).instruction(&I::LocalSet(i32b));
            f.instruction(&I::I32WrapI64).instruction(&I::LocalSet(i32a));
            f.instruction(&I::I32WrapI64).instruction(&I::LocalGet(i32a)).instruction(&I::LocalGet(i32b)).instruction(&ins);
        }
        // init(dst: i64, src: i32, len: i32)
        I::MemoryInit { .. } => wrap_under2(f, &ins, i32a, i32b),

        _ => { f.instruction(&ins); }
    }
    Ok(())
}

pub fn convert(wasm: &[u8]) -> Result<Vec<u8>> {
    let mut type_params: Vec<u32> = Vec::new();
    let mut func_params: Vec<u32> = Vec::new();
    let mut memory64 = false;
    for payload in Parser::new(0).parse_all(wasm) {
        match payload? {
            Payload::TypeSection(s) => {
                for rec in s {
                    for sub in rec?.into_types() {
                        type_params.push(match &sub.composite_type.inner {
                            wasmparser::CompositeInnerType::Func(ft) => ft.params().len() as u32,
                            _ => 0,
                        });
                    }
                }
            }
            Payload::ImportSection(s) => {
                for imp in s.into_imports() {
                    match imp?.ty {
                        wasmparser::TypeRef::Func(t) => func_params.push(type_params[t as usize]),
                        wasmparser::TypeRef::Memory(m) => memory64 |= m.memory64,
                        _ => {}
                    }
                }
            }
            Payload::MemorySection(s) => { for m in s { memory64 |= m?.memory64; } }
            Payload::FunctionSection(s) => { for t in s { func_params.push(type_params[t? as usize]); } }
            _ => {}
        }
    }
    if !memory64 { bail!("game.wasm already uses a 32-bit memory"); }
    let imported = func_params.len() as u32 - Parser::new(0).parse_all(wasm).filter_map(|p| match p { Ok(Payload::FunctionSection(s)) => Some(s.count()), _ => None }).sum::<u32>();
    let mut r = Mem32 { func_params, next_func: imported };
    let mut module = wasm_encoder::Module::new();
    r.parse_core_module(&mut module, Parser::new(0), wasm).map_err(|e| anyhow::anyhow!("convert game.wasm to a 32-bit memory: {e:?}"))?;
    Ok(module.finish())
}
