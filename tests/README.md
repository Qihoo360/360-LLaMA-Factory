# 🧪 Needle Haystack Evaluation Test Configurations

This directory contains comprehensive test configurations for the needle haystack evaluation system, designed to test all possible parameter combinations and features.

## 📁 Directory Structure

```
tests/
├── eval_configs/          # Test YAML configurations
│   ├── test_basic_needle.yaml
│   ├── test_linear_rope.yaml
│   ├── test_yarn_rope.yaml
│   ├── test_longrope_extreme.yaml
│   ├── test_paulgraham_dataset.yaml
│   ├── test_advanced_generation.yaml
│   ├── test_sequence_parallel.yaml
│   └── test_chat_template.yaml
├── run_all_eval_tests.py  # Comprehensive test runner
└── README.md              # This file
```

## 🎯 Test Coverage

### **1. Basic Functionality** - `test_basic_needle.yaml`
- ✅ Basic needle haystack without RoPE scaling
- ✅ Minimal configuration testing
- ✅ Deterministic generation (temp=0.0)
- ✅ Input/output saving

### **2. Linear RoPE Scaling** - `test_linear_rope.yaml`
- ✅ Linear RoPE scaling with 2x factor
- ✅ Model generation config integration
- ✅ Low-temperature creative generation (temp=0.1)
- ✅ Extended context lengths

### **3. YARN RoPE Scaling** - `test_yarn_rope.yaml`
- ✅ YARN RoPE with alpha/beta parameters
- ✅ Multi-GPU sequence parallelism (zigzag-ring)
- ✅ Creative generation (temp=0.3)
- ✅ Nucleus and top-k sampling

### **4. LongRoPE Extreme** - `test_longrope_extreme.yaml`
- ✅ LongRoPE with custom interpolation factors
- ✅ Ulysses sequence parallelism mode
- ✅ Extreme context length scaling (8x)
- ✅ Balanced generation parameters

### **5. PaulGraham Dataset** - `test_paulgraham_dataset.yaml`
- ✅ PaulGraham essays dataset loading
- ✅ Directory-based haystack text
- ✅ Dynamic RoPE scaling
- ✅ Dataset-focused evaluation

### **6. Advanced Generation** - `test_advanced_generation.yaml`
- ✅ All generation parameters testing
- ✅ High creativity (temp=0.7)
- ✅ Advanced sampling techniques
- ✅ Longer response generation

### **7. Sequence Parallelism** - `test_sequence_parallel.yaml`
- ✅ Different parallel modes (llama3)
- ✅ Flash Attention 2 testing
- ✅ Multi-GPU optimization
- ✅ bfloat16 precision

### **8. Chat Template** - `test_chat_template.yaml`
- ✅ Chat template detection and loading
- ✅ Message encoding testing
- ✅ Tokenizer configuration integration
- ✅ Template fallback mechanisms

## 🚀 Running Tests

### **Run All Tests**
```bash
# From repository root
python3 tests/run_all_eval_tests.py
```

### **Run Individual Configurations**
```bash
# Test basic functionality
llamafactory-cli eval tests/eval_configs/test_basic_needle.yaml

# Test YARN RoPE scaling
llamafactory-cli eval tests/eval_configs/test_yarn_rope.yaml

# Test advanced generation
llamafactory-cli eval tests/eval_configs/test_advanced_generation.yaml
```

### **Test Specific Features**
```bash
# Test RoPE scaling variations
llamafactory-cli eval tests/eval_configs/test_linear_rope.yaml
llamafactory-cli eval tests/eval_configs/test_yarn_rope.yaml  
llamafactory-cli eval tests/eval_configs/test_longrope_extreme.yaml

# Test dataset variations
llamafactory-cli eval tests/eval_configs/test_paulgraham_dataset.yaml

# Test parallelism
llamafactory-cli eval tests/eval_configs/test_sequence_parallel.yaml
```

## 📊 Test Results Location

All test results are saved to:
```
tests/results/
├── basic_needle/           # Basic functionality results
├── linear_rope/           # Linear RoPE results  
├── yarn_rope/            # YARN RoPE results
├── longrope_extreme/     # LongRoPE results
├── paulgraham_dataset/   # PaulGraham dataset results
├── advanced_generation/  # Advanced generation results
├── sequence_parallel/    # Sequence parallelism results
└── chat_template/       # Chat template results
```

Each result directory contains:
- `detailed_results.json` - Per-example results
- `summary.json` - Aggregated statistics  
- `inputs_outputs.json` - Input prompts and responses (if enabled)
- Visualization plots (if matplotlib available)

## ✅ Expected Test Outcomes

### **Comprehensive Coverage Testing**
- **RoPE Types**: none, linear, yarn, longrope, dynamic
- **Generation Modes**: deterministic, low-temp, creative, high-temp
- **Datasets**: custom, paulgraham, directory
- **Parallelism**: single-GPU, zigzag-ring, ulysses, llama3
- **Features**: chat templates, generation config, I/O saving

### **Validation Checks**
- ✅ All YAML files parse correctly
- ✅ All required parameters present
- ✅ Parameter combinations work
- ✅ Integration functionality verified
- ✅ Argument parsing successful

## 🎯 Use Cases

### **Development Testing**
- Verify new features work with all configurations
- Test parameter combinations before production
- Validate integration after code changes

### **Performance Benchmarking**  
- Compare different RoPE scaling methods
- Test generation parameter effects
- Evaluate dataset variations

### **Production Validation**
- Verify configurations before deployment
- Test on different hardware setups
- Validate expected functionality

## 💡 Adding New Tests

To add a new test configuration:

1. **Create YAML file** in `tests/eval_configs/`
2. **Follow naming convention**: `test_<feature_name>.yaml`
3. **Include comprehensive parameters** for the feature being tested
4. **Add documentation** explaining what the test validates
5. **Run test suite** to verify integration

## 🔧 Troubleshooting

### **Common Issues**
- **Missing model**: Use lightweight models for testing (DialoGPT-small)
- **GPU memory**: Reduce context lengths or batch size
- **Dataset not found**: Verify PaulGraham essays are available
- **Parse errors**: Validate YAML syntax

### **Debug Mode**
Add verbose logging to any config:
```yaml
# Enable debug output
debug: true
verbose: true
```

The test configurations provide comprehensive coverage of all needle haystack evaluation features and ensure robust, reliable evaluation across different scenarios.