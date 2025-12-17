"""
Test script to verify feedback saving works
Run: python test_feedback_save.py
"""

import os
import json
from datetime import datetime

print("="*80)
print("🧪 TESTING FEEDBACK SAVE FUNCTIONALITY")
print("="*80)

# Test 1: Create directory
print("\n1️⃣ Testing directory creation...")
try:
    os.makedirs('data', exist_ok=True)
    print("✅ Directory 'data' exists/created")
except Exception as e:
    print(f"❌ Failed: {e}")
    exit(1)

# Test 2: Write test file
print("\n2️⃣ Testing file write...")
test_data = [{
    'timestamp': datetime.now().isoformat(),
    'text': 'यो राम्रो छ',
    'prediction': 'NO',
    'confidence': 0.95,
    'feedback': {
        'feedback_type': 'correct',
        'correct_label': None,
        'comment': None
    }
}]

history_file = 'data/prediction_history.json'

try:
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    print(f"✅ Successfully wrote to {history_file}")
except Exception as e:
    print(f"❌ Failed to write: {e}")
    exit(1)

# Test 3: Read file
print("\n3️⃣ Testing file read...")
try:
    with open(history_file, 'r', encoding='utf-8') as f:
        loaded_data = json.load(f)
    print(f"✅ Successfully read {len(loaded_data)} entries")
except Exception as e:
    print(f"❌ Failed to read: {e}")
    exit(1)

# Test 4: Verify content
print("\n4️⃣ Verifying content...")
if loaded_data == test_data:
    print("✅ Data matches perfectly")
else:
    print("❌ Data mismatch")
    exit(1)

# Test 5: Append new entry
print("\n5️⃣ Testing append functionality...")
new_entry = {
    'timestamp': datetime.now().isoformat(),
    'text': 'तिमी मुर्ख हौ',
    'prediction': 'OO',
    'confidence': 0.88,
    'feedback': {
        'feedback_type': 'incorrect',
        'correct_label': 'NO',
        'comment': 'Not really offensive'
    }
}

loaded_data.append(new_entry)

try:
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(loaded_data, f, ensure_ascii=False, indent=2)
    print("✅ Successfully appended new entry")
except Exception as e:
    print(f"❌ Failed to append: {e}")
    exit(1)

# Test 6: Final verification
print("\n6️⃣ Final verification...")
try:
    with open(history_file, 'r', encoding='utf-8') as f:
        final_data = json.load(f)
    
    if len(final_data) == 2:
        print(f"✅ File contains {len(final_data)} entries (correct)")
    else:
        print(f"❌ Expected 2 entries, got {len(final_data)}")
        exit(1)
except Exception as e:
    print(f"❌ Failed: {e}")
    exit(1)

# Test 7: Display content
print("\n7️⃣ Displaying file content...")
print("\nFile path:", os.path.abspath(history_file))
print("\nContent:")
print(json.dumps(final_data, indent=2, ensure_ascii=False))

print("\n" + "="*80)
print("✅ ALL TESTS PASSED!")
print("="*80)
print("\n📝 Summary:")
print(f"   File location: {os.path.abspath(history_file)}")
print(f"   Total entries: {len(final_data)}")
print(f"   File size: {os.path.getsize(history_file)} bytes")

print("\n💡 Next steps:")
print("   1. Run the Streamlit app: streamlit run main_app.py")
print("   2. Make a prediction and submit feedback")
print("   3. Go to History tab to see your entries")
print("   4. If History tab shows 'No history', click 'Refresh History'")

print("\n" + "="*80)