use slotmap::SlotMap;
use crate::values::{ValueKey, ValueData, DefSite};
use crate::operations::{OpKey, OperationData};
use crate::blocks::{BlockKey, BlockData};
use crate::regions::{RegionKey, RegionData};
use crate::functions::FunctionData;
use crate::module::ModuleData;
use crate::types::{TypeId, CoreType, TypeData, TensorTypeInfo, QubitTypeInfo, DataType, Dimension, MemoryLayout};
use crate::attributes::Attributes;
use crate::location::Location;
use crate::interning::{StringId, StringInterner, TypeInterner};
use crate::dialect::{DialectRegistry, register_builtin_dialects};

#[derive(Debug)]
pub struct Context {
    pub values: SlotMap<ValueKey, ValueData>,
    pub ops: SlotMap<OpKey, OperationData>,
    pub blocks: SlotMap<BlockKey, BlockData>,
    pub regions: SlotMap<RegionKey, RegionData>,
    pub strings: StringInterner,
    pub types: TypeInterner,
    pub modules: Vec<ModuleData>,
    pub dialects: DialectRegistry,
}

impl Context {
    pub fn new() -> Self {
        let mut ctx = Self {
            values: SlotMap::with_key(),
            ops: SlotMap::with_key(),
            blocks: SlotMap::with_key(),
            regions: SlotMap::with_key(),
            strings: StringInterner::new(),
            types: TypeInterner::new(),
            modules: Vec::new(),
            dialects: DialectRegistry::new(),
        };
        register_builtin_dialects(&mut ctx.dialects);
        ctx
    }

    // ── String interning ──

    pub fn intern_string(&mut self, s: &str) -> StringId {
        self.strings.intern(s)
    }

    pub fn resolve_string(&self, id: StringId) -> &str {
        self.strings.resolve(id)
    }

    // ── Type interning ──

    pub fn intern_type(&mut self, ty: CoreType) -> TypeId {
        self.types.intern(ty)
    }

    pub fn resolve_type(&self, id: TypeId) -> &CoreType {
        self.types.resolve(id)
    }

    // ── Convenience type constructors ──

    pub fn make_integer_type(&mut self, bits: u32, signed: bool) -> TypeId {
        self.intern_type(CoreType::Integer { bits, signed })
    }

    pub fn make_float_type(&mut self, bits: u32) -> TypeId {
        self.intern_type(CoreType::Float { bits })
    }

    pub fn make_bool_type(&mut self) -> TypeId {
        self.intern_type(CoreType::Boolean)
    }

    pub fn make_void_type(&mut self) -> TypeId {
        self.intern_type(CoreType::Void)
    }

    pub fn make_index_type(&mut self) -> TypeId {
        self.intern_type(CoreType::Index)
    }

    pub fn make_tensor_type(&mut self, shape: Vec<Dimension>, dtype: DataType, layout: MemoryLayout) -> TypeId {
        let dialect = self.intern_string("tensor");
        let name = self.intern_string("tensor");
        self.intern_type(CoreType::Opaque {
            dialect,
            name,
            data: TypeData::Tensor(TensorTypeInfo { shape, dtype, layout }),
        })
    }

    pub fn make_qubit_type(&mut self) -> TypeId {
        let dialect = self.intern_string("quantum");
        let name = self.intern_string("qubit");
        self.intern_type(CoreType::Opaque {
            dialect,
            name,
            data: TypeData::Qubit(QubitTypeInfo::Logical),
        })
    }

    pub fn make_physical_qubit_type(&mut self, id: usize, t1: f64, t2: f64, freq: f64, fidelity: f64) -> TypeId {
        let dialect = self.intern_string("quantum");
        let name = self.intern_string("physical_qubit");
        use crate::types::OrderedFloat;
        self.intern_type(CoreType::Opaque {
            dialect,
            name,
            data: TypeData::Qubit(QubitTypeInfo::Physical {
                id,
                t1_us: OrderedFloat::from_f64(t1),
                t2_us: OrderedFloat::from_f64(t2),
                freq_ghz: OrderedFloat::from_f64(freq),
                fidelity: OrderedFloat::from_f64(fidelity),
            }),
        })
    }

    pub fn make_bit_type(&mut self) -> TypeId {
        let dialect = self.intern_string("quantum");
        let name = self.intern_string("bit");
        self.intern_type(CoreType::Opaque {
            dialect,
            name,
            data: TypeData::ClassicalBit,
        })
    }

    pub fn make_hamiltonian_type(&mut self, num_qubits: usize) -> TypeId {
        let dialect = self.intern_string("quantum");
        let name = self.intern_string("hamiltonian");
        self.intern_type(CoreType::Opaque {
            dialect,
            name,
            data: TypeData::Hamiltonian { num_qubits },
        })
    }

    pub fn make_function_type(&mut self, params: Vec<TypeId>, returns: Vec<TypeId>) -> TypeId {
        self.intern_type(CoreType::Function { params, returns })
    }

    pub fn make_tuple_type(&mut self, elems: Vec<TypeId>) -> TypeId {
        self.intern_type(CoreType::Tuple(elems))
    }

    // ── Value creation ──

    pub fn create_value(&mut self, ty: TypeId, name: Option<StringId>, def: DefSite) -> ValueKey {
        self.values.insert(ValueData { ty, name, def })
    }

    pub fn get_value(&self, key: ValueKey) -> Option<&ValueData> {
        self.values.get(key)
    }

    pub fn value_type(&self, key: ValueKey) -> Option<TypeId> {
        self.values.get(key).map(|v| v.ty)
    }

    // ── Operation creation ──

    pub fn create_op(
        &mut self,
        name: &str,
        dialect: &str,
        inputs: Vec<ValueKey>,
        result_types: Vec<TypeId>,
        attrs: Attributes,
        location: Location,
    ) -> (OpKey, Vec<ValueKey>) {
        let name_id = self.intern_string(name);
        let dialect_id = self.intern_string(dialect);

        let op_key = self.ops.insert(OperationData {
            name: name_id,
            dialect: dialect_id,
            inputs,
            results: Vec::new(),
            attrs,
            regions: Vec::new(),
            location,
            parent_block: None,
        });

        let mut result_keys = Vec::with_capacity(result_types.len());
        for (i, ty) in result_types.iter().enumerate() {
            let val_key = self.create_value(
                *ty,
                None,
                DefSite::OpResult { op: op_key, result_index: i as u32 },
            );
            result_keys.push(val_key);
        }

        self.ops[op_key].results = result_keys.clone();
        (op_key, result_keys)
    }

    pub fn get_op(&self, key: OpKey) -> Option<&OperationData> {
        self.ops.get(key)
    }

    pub fn get_op_mut(&mut self, key: OpKey) -> Option<&mut OperationData> {
        self.ops.get_mut(key)
    }

    pub fn op_name(&self, key: OpKey) -> &str {
        let op = &self.ops[key];
        self.strings.resolve(op.name)
    }

    pub fn op_dialect(&self, key: OpKey) -> &str {
        let op = &self.ops[key];
        self.strings.resolve(op.dialect)
    }

    // ── Block creation ──

    pub fn create_block(&mut self) -> BlockKey {
        self.blocks.insert(BlockData::new())
    }

    pub fn create_block_arg(&mut self, block: BlockKey, ty: TypeId) -> ValueKey {
        let arg_index = self.blocks[block].args.len() as u32;
        let val_key = self.create_value(
            ty,
            None,
            DefSite::BlockArg { block, arg_index },
        );
        self.blocks[block].args.push(val_key);
        val_key
    }

    pub fn add_op_to_block(&mut self, block: BlockKey, op: OpKey) {
        self.blocks[block].ops.push(op);
        self.ops[op].parent_block = Some(block);
    }

    pub fn get_block(&self, key: BlockKey) -> Option<&BlockData> {
        self.blocks.get(key)
    }

    // ── Region creation ──

    pub fn create_region(&mut self) -> RegionKey {
        self.regions.insert(RegionData::new())
    }

    pub fn add_block_to_region(&mut self, region: RegionKey, block: BlockKey) {
        if self.regions[region].entry_block.is_none() {
            self.regions[region].entry_block = Some(block);
        }
        self.regions[region].blocks.push(block);
        self.blocks[block].parent_region = Some(region);
    }

    pub fn attach_region_to_op(&mut self, op: OpKey, region: RegionKey) {
        self.ops[op].regions.push(region);
        self.regions[region].parent_op = Some(op);
    }

    pub fn get_region(&self, key: RegionKey) -> Option<&RegionData> {
        self.regions.get(key)
    }

    // ── Module creation ──

    pub fn create_module(&mut self, name: &str) -> usize {
        let name_id = self.intern_string(name);
        let module = ModuleData::new(name_id);
        self.modules.push(module);
        self.modules.len() - 1
    }

    pub fn get_module(&self, index: usize) -> Option<&ModuleData> {
        self.modules.get(index)
    }

    pub fn get_module_mut(&mut self, index: usize) -> Option<&mut ModuleData> {
        self.modules.get_mut(index)
    }

    pub fn add_function_to_module(&mut self, module_index: usize, func: FunctionData) {
        if let Some(module) = self.modules.get_mut(module_index) {
            module.add_function(func);
        }
    }

    // ── Snapshot/restore for pass rollback ──

    pub fn snapshot(&self) -> ContextSnapshot {
        ContextSnapshot {
            num_values: self.values.len(),
            num_ops: self.ops.len(),
            num_blocks: self.blocks.len(),
            num_regions: self.regions.len(),
        }
    }

    // ── Type queries ──

    pub fn is_qubit_type(&self, ty: TypeId) -> bool {
        matches!(self.resolve_type(ty), CoreType::Opaque { data: TypeData::Qubit(_), .. })
    }

    pub fn is_tensor_type(&self, ty: TypeId) -> bool {
        matches!(self.resolve_type(ty), CoreType::Opaque { data: TypeData::Tensor(_), .. })
    }

    pub fn is_bit_type(&self, ty: TypeId) -> bool {
        matches!(self.resolve_type(ty), CoreType::Opaque { data: TypeData::ClassicalBit, .. })
    }

    pub fn get_tensor_info(&self, ty: TypeId) -> Option<&TensorTypeInfo> {
        match self.resolve_type(ty) {
            CoreType::Opaque { data: TypeData::Tensor(info), .. } => Some(info),
            _ => None,
        }
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct ContextSnapshot {
    pub num_values: usize,
    pub num_ops: usize,
    pub num_blocks: usize,
    pub num_regions: usize,
}
