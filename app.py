import pandas as pd
import json
import re

def extract_complete_pipeline_flows(csv_file_path):
    # Read the pipeline CSV file
    df_pipelines = pd.read_csv(csv_file_path)
    
    print("🔄 Reading pipeline data...")
    print(f"Total rows in CSV: {len(df_pipelines)}")
    print(f"Unique pipeline tags: {len(df_pipelines['Tag'].unique())}")
    
    # Clean the data
    df_pipelines['From'] = df_pipelines['From'].astype(str).str.strip()
    df_pipelines['To'] = df_pipelines['To'].astype(str).str.strip()
    df_pipelines['Tag'] = df_pipelines['Tag'].astype(str).str.strip()
    
    # Remove problematic characters
    df_pipelines['From'] = df_pipelines['From'].apply(clean_text)
    df_pipelines['To'] = df_pipelines['To'].apply(clean_text)
    
    # Dictionary to store all pipeline flows
    pipeline_flows = {}
    
    # Process each pipeline tag
    for tag in df_pipelines['Tag'].unique():
        if pd.isna(tag) or not tag or tag == 'nan':
            continue
            
        print(f"\n🔍 Processing pipeline: {tag}")
        
        # Get ALL rows for this pipeline tag
        tag_data = df_pipelines[df_pipelines['Tag'] == tag]
        
        # Extract ALL connections in order
        connections = []
        for _, row in tag_data.iterrows():
            from_node = str(row['From']).strip()
            to_node = str(row['To']).strip()
            
            if from_node and from_node != 'nan' and to_node and to_node != 'nan':
                connections.append({
                    'from': from_node,
                    'to': to_node
                })
        
        print(f"   Found {len(connections)} connections")
        
        if connections:
            # Build complete flow using ALL connections
            complete_flow = build_complete_flow_from_all_connections(connections)
            
            pipeline_flows[tag] = {
                'pipeline_tag': tag,
                'total_connections': len(connections),
                'complete_flow': complete_flow,
                'start': complete_flow[0] if complete_flow else None,
                'end': complete_flow[-1] if complete_flow else None,
                'all_raw_connections': connections  # Keep all original connections for verification
            }
            
            print(f"   ✅ Complete flow: {len(complete_flow)} nodes")
            print(f"   Start: {complete_flow[0] if complete_flow else 'None'}")
            print(f"   End: {complete_flow[-1] if complete_flow else 'None'}")
        else:
            print(f"   ❌ No valid connections found")
    
    return pipeline_flows

def build_complete_flow_from_all_connections(connections):
    """Build complete flow path using ALL connections in sequential order"""
    if not connections:
        return []
    
    # Start with the first connection
    flow_nodes = []
    
    # Add all unique nodes in the order they appear in connections
    for conn in connections:
        from_node = conn['from']
        to_node = conn['to']
        
        # Add from_node if not already in flow
        if from_node not in flow_nodes:
            flow_nodes.append(from_node)
        
        # Add to_node if not already in flow  
        if to_node not in flow_nodes:
            flow_nodes.append(to_node)
    
    # Now try to reorder based on connection sequence
    ordered_flow = []
    
    if flow_nodes:
        # Start with first node from first connection
        ordered_flow = [connections[0]['from']]
        
        # Keep building the sequence by following connections
        changed = True
        while changed and len(ordered_flow) < len(flow_nodes):
            changed = False
            
            # Try to extend from the end
            last_node = ordered_flow[-1]
            for conn in connections:
                if conn['from'] == last_node and conn['to'] not in ordered_flow:
                    ordered_flow.append(conn['to'])
                    changed = True
                    break
            
            # If couldn't extend from end, try to prepend from start
            if not changed:
                first_node = ordered_flow[0]
                for conn in connections:
                    if conn['to'] == first_node and conn['from'] not in ordered_flow:
                        ordered_flow.insert(0, conn['from'])
                        changed = True
                        break
        
        # If we still don't have all nodes, add the remaining ones
        if len(ordered_flow) < len(flow_nodes):
            remaining_nodes = [node for node in flow_nodes if node not in ordered_flow]
            ordered_flow.extend(remaining_nodes)
    
    return ordered_flow

def clean_text(text):
    """Clean text from encoding issues"""
    if pd.isna(text) or text == 'nan':
        return ""
    
    text = str(text)
    # Remove encoding artifacts but keep the structure
    text = re.sub(r'Âḟ', ' ', text)
    text = re.sub(r'Â', ' ', text)
    text = re.sub(r'Ḟ', ' ', text)
    text = re.sub(r'±', ' ', text)
    text = re.sub(r'°', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    return text.strip()

def save_flows_to_json(pipeline_flows, output_file):
    """Save the pipeline flows to JSON file"""
    # Convert to simple JSON structure for this step
    json_output = {}
    
    for pipeline_tag, data in pipeline_flows.items():
        json_output[pipeline_tag] = {
            "pipeline_tag": pipeline_tag,
            "start": data['start'],
            "end": data['end'],
            "complete_flow": data['complete_flow'],
            "total_connections": data['total_connections'],
            "flow_length": len(data['complete_flow']),
            "all_connections": data['all_raw_connections']
        }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Pipeline flows saved to: {output_file}")
    return json_output

# File path
csv_file_path = "output/csv_output/PipeLines.csv"

print("🚀 EXTRACTING COMPLETE PIPELINE FLOWS")
print("=" * 60)

try:
    # Step 1: Extract complete flows
    pipeline_flows = extract_complete_pipeline_flows(csv_file_path)
    
    # Step 2: Save to JSON
    output_file = "complete_pipeline_flows_step1.json"
    json_result = save_flows_to_json(pipeline_flows, output_file)
    
    # Step 3: Show summary
    print(f"\n📊 EXTRACTION SUMMARY:")
    print("=" * 50)
    print(f"Total pipelines processed: {len(pipeline_flows)}")
    
    total_connections = sum(flow['total_connections'] for flow in pipeline_flows.values())
    total_nodes = sum(len(flow['complete_flow']) for flow in pipeline_flows.values())
    
    print(f"Total connections: {total_connections}")
    print(f"Total nodes in all flows: {total_nodes}")
    
    # Show details for each pipeline
    print(f"\n🔍 PIPELINE DETAILS:")
    for pipeline_tag, data in pipeline_flows.items():
        print(f"\nPipeline: {pipeline_tag}")
        print(f"  Connections: {data['total_connections']}")
        print(f"  Flow length: {len(data['complete_flow'])}")
        print(f"  Start: {data['start']}")
        print(f"  End: {data['end']}")
        
        # Show all connections for verification
        print(f"  All connections:")
        for i, conn in enumerate(data['all_raw_connections'], 1):
            print(f"    {i}. {conn['from']} → {conn['to']}")
            
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()