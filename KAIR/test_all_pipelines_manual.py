#!/usr/bin/env python3
"""
Interactive Manual Model Testing CLI
===================================

An interactive command-line interface for testing individual models,
specific datasets, and resuming/managing inference sessions.

Features:
- Interactive model selection (conv_nstb, conv_trans_nstb, trans_nstb)
- Dataset selection (original, clinical, both)
- Inference session management (new, resume, continue)
- Custom output folder naming
- Real-time progress monitoring
- Detailed logging and error handling

Usage:
    python test_all_pipelines_manual.py
"""

import os
import sys
import json
import time
import glob
from pathlib import Path
from datetime import datetime
import torch
import traceback

# Add KAIR to path for imports
sys.path.append('/home/grad/mppatel/Documents/Project/SCUNet-NGSWIN/KAIR')

# Import organized utility modules
from inference_utils.model_utils import (
    discover_all_trained_models, 
    discover_all_trained_models_with_checkpoints,
    setup_model_and_checkpoint, 
    setup_model_and_checkpoint_with_selection,
    get_available_checkpoints,
    get_next_inference_folder,
    get_smart_remaining_models,
    check_folder_completely_finished
)
from inference_utils.dataset_processing import (
    process_original_dataset, 
    process_clinical_dataset_no_masks,
    process_clinical_artifact_only_dataset
)
from inference_utils.visualization import create_comprehensive_analysis

class ResultsConfig:
    """Configuration class for controlling results output"""
    
    def __init__(self):
        # Image saving frequency controls
        self.save_every_nth = 15  # Save comparison images every N samples
        self.preprocessing_every_nth = 30  # Save preprocessing visualizations every N samples
        self.detailed_every_nth = 45  # Save detailed analysis every N samples
        
        # Limits on saved results
        self.max_comparisons = 500  # Maximum comparison images to save
        self.max_preprocessing_vis = 250  # Maximum preprocessing visualizations
        self.max_detailed_analysis = 100  # Maximum detailed analysis images
        
        # Important results control
        self.save_best_samples = 10  # Number of best PSNR/SSIM samples to save
        self.save_worst_samples = 5   # Number of worst PSNR/SSIM samples to save
        
        # Analysis and metrics control
        self.create_metrics_plots = True  # Create distribution plots for metrics
        self.save_individual_metrics = True  # Save per-sample metrics
        self.create_comprehensive_analysis = True  # Create final comprehensive analysis
        
        # Storage and format options
        self.image_format = 'png'  # Image format: png, jpg, tiff
        self.image_quality = 95   # For jpg format (1-100)
        self.image_dpi = 150      # DPI for saved images
        self.compress_results = False  # Compress results folder after completion
        
        # Clinical dataset specific controls
        self.clinical_sample_limit = None  # Limit processing to N slices (None = no limit)
        self.clinical_save_frequency = 20  # Save every N slices for clinical data
        
        # Memory and performance
        self.batch_process = False  # Process in batches to save memory
        self.clear_cache_frequency = 100  # Clear CUDA cache every N samples
        
        # Debugging and logging
        self.verbose_logging = True  # Detailed logging
        self.save_processing_logs = True  # Save detailed processing logs
        self.show_progress_bar = True  # Show progress bars during processing
        
    def to_dict(self):
        """Convert config to dictionary for saving"""
        return {k: v for k, v in self.__dict__.items()}
        
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary"""
        config = cls()
        for k, v in config_dict.items():
            if hasattr(config, k):
                setattr(config, k, v)
        return config
        
    def get_preset(self, preset_name):
        """Get predefined configuration presets"""
        presets = {
            'minimal': {
                'save_every_nth': 50,
                'preprocessing_every_nth': 100,
                'detailed_every_nth': 150,
                'max_comparisons': 50,
                'max_preprocessing_vis': 25,
                'max_detailed_analysis': 20,
                'save_best_samples': 5,
                'save_worst_samples': 3,
                'create_metrics_plots': True,
                'create_comprehensive_analysis': True,
                'clinical_save_frequency': 50
            },
            'standard': {
                'save_every_nth': 15,
                'preprocessing_every_nth': 30,
                'detailed_every_nth': 45,
                'max_comparisons': 200,
                'max_preprocessing_vis': 100,
                'max_detailed_analysis': 50,
                'save_best_samples': 10,
                'save_worst_samples': 5,
                'create_metrics_plots': True,
                'create_comprehensive_analysis': True,
                'clinical_save_frequency': 20
            },
            'comprehensive': {
                'save_every_nth': 5,
                'preprocessing_every_nth': 10,
                'detailed_every_nth': 15,
                'max_comparisons': 1000,
                'max_preprocessing_vis': 500,
                'max_detailed_analysis': 200,
                'save_best_samples': 20,
                'save_worst_samples': 10,
                'create_metrics_plots': True,
                'create_comprehensive_analysis': True,
                'clinical_save_frequency': 10
            },
            'fast': {
                'save_every_nth': 100,
                'preprocessing_every_nth': 200,
                'detailed_every_nth': 300,
                'max_comparisons': 20,
                'max_preprocessing_vis': 10,
                'max_detailed_analysis': 5,
                'save_best_samples': 3,
                'save_worst_samples': 2,
                'create_metrics_plots': False,
                'create_comprehensive_analysis': True,
                'clinical_save_frequency': 100
            }
        }
        
        if preset_name in presets:
            for k, v in presets[preset_name].items():
                setattr(self, k, v)
        return self

class InteractiveCLI:
    """Interactive CLI for model testing"""
    
    def __init__(self):
        self.base_output = '/home/grad/mppatel/Documents/Project/SCUNet-NGSWIN/KAIR/inference_results'
        self.default_paths = {
            'original_data': '/home/Drive-D/SynDeepLesion/test_640geo',
            'clinical_data': '/home/Drive-D/clinical_metal',
            'clinical_masks': '/home/Drive-D/clinical_metal_mask',
            'clinical_artifact_only': '/home/Drive-D/clincial_metal_test_Metal_artifact_only/test'
        }
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results_config = ResultsConfig()  # Default configuration
        
    def print_header(self):
        """Print welcome header"""
        print("\n" + "="*80)
        print("🚀 INTERACTIVE MODEL TESTING CLI")
        print("="*80)
        print("Welcome to the manual model testing interface!")
        print(f"🔧 Device: {self.device}")
        print(f"📂 Base output: {self.base_output}")
        print("="*80)
        
    def print_menu(self, options, title="Select an option"):
        """Print numbered menu options"""
        print(f"\n📋 {title}:")
        print("-" * (len(title) + 5))
        for i, option in enumerate(options, 1):
            print(f"  {i}. {option}")
        print(f"  0. Exit")
        
    def get_user_choice(self, max_option, prompt="Enter your choice"):
        """Get valid user input"""
        while True:
            try:
                choice = input(f"\n{prompt} (0-{max_option}): ").strip()
                choice_int = int(choice)
                if 0 <= choice_int <= max_option:
                    return choice_int
                else:
                    print(f"❌ Please enter a number between 0 and {max_option}")
            except ValueError:
                print("❌ Please enter a valid number")
                
    def get_yes_no(self, prompt):
        """Get yes/no input from user"""
        while True:
            response = input(f"{prompt} (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("❌ Please enter 'y' or 'n'")
                
    def discover_and_select_models(self):
        """Discover available models and let user select"""
        print("\n🔍 Discovering available trained models...")
        models_info = discover_all_trained_models_with_checkpoints()
        
        if not models_info:
            print("❌ No trained models found! Check training_results directory.")
            return None
            
        print(f"✓ Found {len(models_info)} trained models")
        
        # Add option to select all models
        model_options = [f"{model['variant']} - {model['display_name']}" for model in models_info]
        model_options.append("ALL MODELS (run all variants)")
        
        self.print_menu(model_options, "Select model(s) to test")
        choice = self.get_user_choice(len(model_options))
        
        if choice == 0:
            return None
        elif choice == len(model_options):  # ALL MODELS
            selected_models = models_info
        else:
            selected_models = [models_info[choice - 1]]
        
        # Now select checkpoints for each model
        final_models = []
        for model in selected_models:
            checkpoint_selection = self.select_checkpoint_for_model(model)
            if checkpoint_selection:
                model['selected_checkpoint'] = checkpoint_selection
                final_models.append(model)
        
        return final_models if final_models else None
        
    def select_checkpoint_for_model(self, model_info):
        """Select checkpoint for a specific model"""
        print(f"\n🎯 Select checkpoint for {model_info['variant'].upper()}")
        print("-" * 50)
        
        checkpoints = model_info['checkpoints']
        options = []
        checkpoint_map = {}
        
        # Add PSNR checkpoint if available
        if checkpoints['psnr']:
            options.append(f"🏆 Best PSNR - {checkpoints['psnr']['date']}")
            checkpoint_map[len(options)] = checkpoints['psnr']
        
        # Add latest checkpoint if different from PSNR
        if checkpoints['latest'] and (not checkpoints['psnr'] or 
            checkpoints['latest']['iteration'] != checkpoints['psnr'].get('iteration', 'best')):
            options.append(f"🕐 Latest - Iteration {checkpoints['latest']['iteration']:,} ({checkpoints['latest']['date']})")
            checkpoint_map[len(options)] = checkpoints['latest']
        
        # Add option to browse all iterations
        if checkpoints['iterations']:
            options.append(f"📋 Browse all {len(checkpoints['iterations'])} iterations")
            
        # Add option to use default
        default_type = "PSNR" if checkpoints['psnr'] else "Latest"
        options.append(f"⚡ Use default ({default_type})")
        
        self.print_menu(options, f"Checkpoint options for {model_info['variant']}")
        choice = self.get_user_choice(len(options))
        
        if choice == 0:
            return None
        elif choice in checkpoint_map:
            return checkpoint_map[choice]
        elif choice == len(options):  # Use default
            return model_info['default_checkpoint']
        elif choice == len(options) - 1:  # Browse all iterations
            return self.browse_all_iterations(model_info)
        
        return None
        
    def browse_all_iterations(self, model_info):
        """Browse and select from all available iterations"""
        iterations = model_info['checkpoints']['iterations']
        
        if not iterations:
            print("❌ No iteration checkpoints available")
            return None
        
        print(f"\n📊 All available iterations for {model_info['variant'].upper()}")
        print("=" * 60)
        
        # Group iterations for better display
        iterations_per_page = 15
        total_pages = (len(iterations) + iterations_per_page - 1) // iterations_per_page
        current_page = 0
        
        while True:
            start_idx = current_page * iterations_per_page
            end_idx = min(start_idx + iterations_per_page, len(iterations))
            page_iterations = iterations[start_idx:end_idx]
            
            print(f"\nPage {current_page + 1}/{total_pages} (Iterations {start_idx + 1}-{end_idx} of {len(iterations)})")
            print("-" * 60)
            
            options = []
            for i, iteration in enumerate(page_iterations):
                display_text = f"Iteration {iteration['iteration']:>7,} - {iteration['date']}"
                options.append(display_text)
            
            # Navigation options
            nav_options = []
            if current_page > 0:
                nav_options.append("◀️  Previous page")
            if current_page < total_pages - 1:
                nav_options.append("▶️  Next page")
            nav_options.extend([
                "🔍 Search by iteration number",
                "🏆 Use best PSNR instead",
                "🕐 Use latest iteration",
                "↩️  Back to checkpoint selection"
            ])
            
            all_options = options + [""] + nav_options  # Empty string for separator
            
            print("Available iterations:")
            for i, option in enumerate(options, 1):
                print(f"  {i:2d}. {option}")
            
            print("\nNavigation:")
            for i, option in enumerate(nav_options, len(options) + 1):
                if option == "":
                    continue
                print(f"  {i:2d}. {option}")
            
            choice = self.get_user_choice(len(options) + len(nav_options))
            
            if choice == 0:
                return None
            elif choice <= len(options):
                # Selected an iteration
                return page_iterations[choice - 1]
            else:
                # Navigation option
                nav_choice = choice - len(options) - 1
                nav_option = nav_options[nav_choice]
                
                if "Previous page" in nav_option:
                    current_page -= 1
                elif "Next page" in nav_option:
                    current_page += 1
                elif "Search by iteration" in nav_option:
                    return self.search_iteration(iterations)
                elif "Use best PSNR" in nav_option:
                    return model_info['checkpoints']['psnr']
                elif "Use latest" in nav_option:
                    return model_info['checkpoints']['latest']
                elif "Back to checkpoint" in nav_option:
                    return self.select_checkpoint_for_model(model_info)
    
    def search_iteration(self, iterations):
        """Search for specific iteration number"""
        while True:
            try:
                target = input("\n🔍 Enter iteration number to search for: ").strip()
                if not target:
                    return None
                
                target_num = int(target)
                
                # Find closest iterations
                exact_match = None
                closest_iterations = []
                
                for iteration in iterations:
                    if iteration['iteration'] == target_num:
                        exact_match = iteration
                        break
                    else:
                        diff = abs(iteration['iteration'] - target_num)
                        closest_iterations.append((diff, iteration))
                
                if exact_match:
                    print(f"✅ Found exact match: Iteration {exact_match['iteration']:,}")
                    return exact_match
                else:
                    # Show closest matches
                    closest_iterations.sort(key=lambda x: x[0])
                    closest_5 = closest_iterations[:5]
                    
                    print(f"\n❌ Iteration {target_num:,} not found. Closest matches:")
                    options = []
                    for diff, iteration in closest_5:
                        display_text = f"Iteration {iteration['iteration']:,} (±{diff:,}) - {iteration['date']}"
                        options.append(display_text)
                        print(f"   • {display_text}")
                    
                    print(f"\nSelect from closest matches or try again:")
                    for i, (diff, iteration) in enumerate(closest_5, 1):
                        print(f"  {i}. Iteration {iteration['iteration']:,}")
                    print(f"  {len(closest_5) + 1}. Search again")
                    print(f"  0. Cancel")
                    
                    choice = self.get_user_choice(len(closest_5) + 1)
                    if choice == 0:
                        return None
                    elif choice <= len(closest_5):
                        return closest_5[choice - 1][1]
                    # else continue loop for search again
                        
            except ValueError:
                print("❌ Please enter a valid number")
            
    def select_datasets(self):
        """Let user select which datasets to test"""
        dataset_options = [
            "Original dataset only (test_640geo)",
            "Clinical dataset only (no masks)", 
            "Clinical artifact-only dataset (test set)",
            "Original + Clinical datasets",
            "Original + Clinical artifact-only datasets", 
            "Clinical + Clinical artifact-only datasets",
            "All three datasets",
            "Custom dataset paths"
        ]
        
        self.print_menu(dataset_options, "Select dataset(s) to test")
        choice = self.get_user_choice(len(dataset_options))
        
        if choice == 0:
            return None
            
        datasets = {
            'test_original': False,
            'test_clinical': False,
            'test_clinical_artifact_only': False,
            'original_path': self.default_paths['original_data'],
            'clinical_path': self.default_paths['clinical_data'],
            'clinical_artifact_only_path': self.default_paths['clinical_artifact_only']
        }
        
        if choice == 1:  # Original only
            datasets['test_original'] = True
        elif choice == 2:  # Clinical only
            datasets['test_clinical'] = True
        elif choice == 3:  # Clinical artifact-only only
            datasets['test_clinical_artifact_only'] = True
        elif choice == 4:  # Original + Clinical
            datasets['test_original'] = True
            datasets['test_clinical'] = True
        elif choice == 5:  # Original + Clinical artifact-only
            datasets['test_original'] = True
            datasets['test_clinical_artifact_only'] = True
        elif choice == 6:  # Clinical + Clinical artifact-only
            datasets['test_clinical'] = True
            datasets['test_clinical_artifact_only'] = True
        elif choice == 7:  # All three
            datasets['test_original'] = True
            datasets['test_clinical'] = True
            datasets['test_clinical_artifact_only'] = True
        elif choice == 8:  # Custom paths
            datasets = self.get_custom_paths()
            if not datasets:
                return None
                
        return datasets
        
    def get_custom_paths(self):
        """Get custom dataset paths from user"""
        print("\n📁 Enter custom dataset paths:")
        
        datasets = {
            'test_original': False, 
            'test_clinical': False, 
            'test_clinical_artifact_only': False
        }
        
        # Original dataset
        if self.get_yes_no("🔍 Test original dataset?"):
            while True:
                path = input(f"📂 Original dataset path [{self.default_paths['original_data']}]: ").strip()
                if not path:
                    path = self.default_paths['original_data']
                if os.path.exists(path):
                    datasets['original_path'] = path
                    datasets['test_original'] = True
                    break
                else:
                    print(f"❌ Path not found: {path}")
                    if not self.get_yes_no("Try again?"):
                        break
                        
        # Clinical dataset  
        if self.get_yes_no("🏥 Test clinical dataset?"):
            while True:
                path = input(f"📂 Clinical dataset path [{self.default_paths['clinical_data']}]: ").strip()
                if not path:
                    path = self.default_paths['clinical_data']
                if os.path.exists(path):
                    datasets['clinical_path'] = path
                    datasets['test_clinical'] = True
                    break
                else:
                    print(f"❌ Path not found: {path}")
                    if not self.get_yes_no("Try again?"):
                        break

        # Clinical artifact-only dataset  
        if self.get_yes_no("🧪 Test clinical artifact-only dataset?"):
            while True:
                path = input(f"📂 Clinical artifact-only path [{self.default_paths['clinical_artifact_only']}]: ").strip()
                if not path:
                    path = self.default_paths['clinical_artifact_only']
                if os.path.exists(path):
                    datasets['clinical_artifact_only_path'] = path
                    datasets['test_clinical_artifact_only'] = True
                    break
                else:
                    print(f"❌ Path not found: {path}")
                    if not self.get_yes_no("Try again?"):
                        break
                        
        if not any([datasets['test_original'], datasets['test_clinical'], datasets['test_clinical_artifact_only']]):
            print("❌ No datasets selected!")
            return None
            
        return datasets
        
    def manage_output_folder(self):
        """Manage output folder selection and resume options"""
        # Check existing inference folders
        existing_folders = sorted(glob.glob(os.path.join(self.base_output, "inference_*")))
        
        if existing_folders:
            print(f"\n📂 Found {len(existing_folders)} existing inference folder(s):")
            for folder in existing_folders:
                folder_name = os.path.basename(folder)
                status = "✅ Complete" if check_folder_completely_finished(folder) else "⏳ Incomplete"
                print(f"   {folder_name}: {status}")
                
            folder_options = [
                "Create new inference folder",
                "Resume incomplete work (if any)",
                "Continue in specific folder",
                "Custom folder name"
            ]
            
            self.print_menu(folder_options, "Choose output folder option")
            choice = self.get_user_choice(len(folder_options))
            
            if choice == 0:
                return None
            elif choice == 1:  # New folder
                return self.create_new_folder()
            elif choice == 2:  # Resume incomplete
                return self.resume_incomplete_work(existing_folders)
            elif choice == 3:  # Specific folder
                return self.select_specific_folder(existing_folders)
            elif choice == 4:  # Custom name
                return self.create_custom_folder()
        else:
            print("\n📂 No existing inference folders found.")
            if self.get_yes_no("Create first inference folder?"):
                return self.create_new_folder()
            else:
                return None
                
    def create_new_folder(self):
        """Create new inference folder"""
        output_dir = get_next_inference_folder(self.base_output)
        print(f"📁 Created new folder: {os.path.basename(output_dir)}")
        return {'path': output_dir, 'mode': 'new', 'models': None}
        
    def resume_incomplete_work(self, existing_folders):
        """Resume incomplete work"""
        incomplete_folders = [f for f in existing_folders if not check_folder_completely_finished(f)]
        
        if not incomplete_folders:
            print("✅ All existing folders are complete!")
            if self.get_yes_no("Create new folder instead?"):
                return self.create_new_folder()
            else:
                return None
                
        if len(incomplete_folders) == 1:
            folder = incomplete_folders[0]
            print(f"🔄 Resuming work in: {os.path.basename(folder)}")
        else:
            print(f"\n🔍 Found {len(incomplete_folders)} incomplete folder(s):")
            folder_options = [os.path.basename(f) for f in incomplete_folders]
            self.print_menu(folder_options, "Select folder to resume")
            choice = self.get_user_choice(len(folder_options))
            if choice == 0:
                return None
            folder = incomplete_folders[choice - 1]
            
        # Get remaining models for this folder
        remaining_models = get_smart_remaining_models(folder)
        return {'path': folder, 'mode': 'resume', 'models': remaining_models}
        
    def select_specific_folder(self, existing_folders):
        """Select specific folder to continue work"""
        folder_options = [os.path.basename(f) for f in existing_folders]
        self.print_menu(folder_options, "Select folder to use")
        choice = self.get_user_choice(len(folder_options))
        
        if choice == 0:
            return None
            
        folder = existing_folders[choice - 1]
        print(f"📁 Selected folder: {os.path.basename(folder)}")
        
        mode = 'overwrite' if self.get_yes_no("⚠️  Overwrite existing results?") else 'continue'
        return {'path': folder, 'mode': mode, 'models': None}
        
    def create_custom_folder(self):
        """Create folder with custom name"""
        while True:
            name = input("📝 Enter custom folder name (e.g., 'test_experiment_1'): ").strip()
            if not name:
                print("❌ Please enter a valid name")
                continue
                
            if not name.startswith('inference_'):
                name = f"inference_{name}"
                
            custom_path = os.path.join(self.base_output, name)
            
            if os.path.exists(custom_path):
                print(f"⚠️  Folder {name} already exists!")
                if self.get_yes_no("Use existing folder?"):
                    return {'path': custom_path, 'mode': 'continue', 'models': None}
                else:
                    continue
            else:
                os.makedirs(custom_path, exist_ok=True)
                print(f"📁 Created custom folder: {name}")
                return {'path': custom_path, 'mode': 'new', 'models': None}
                
    def show_test_summary(self, models, datasets, output_info):
        """Show summary of what will be tested"""
        print("\n" + "="*60)
        print("📋 TEST SUMMARY")
        print("="*60)
        
        print(f"🎯 Models to test ({len(models)}):")
        for model in models:
            checkpoint_info = model.get('selected_checkpoint', model['default_checkpoint'])
            checkpoint_desc = f"{checkpoint_info['display_name']} ({checkpoint_info['date']})"
            print(f"   • {model['variant']} - {checkpoint_desc}")
            
        print(f"\n📊 Datasets to test:")
        if datasets['test_original']:
            print(f"   • Original: {datasets['original_path']}")
        if datasets['test_clinical']:
            print(f"   • Clinical: {datasets['clinical_path']}")
        if datasets['test_clinical_artifact_only']:
            print(f"   • Clinical Artifact-Only: {datasets['clinical_artifact_only_path']}")
            
        print(f"\n📂 Output folder:")
        print(f"   • Path: {os.path.basename(output_info['path'])}")
        print(f"   • Mode: {output_info['mode']}")
        
        if output_info['mode'] == 'resume' and output_info['models']:
            print(f"   • Resuming {len(output_info['models'])} remaining models")
            
        print("="*60)
        
        return self.get_yes_no("🚀 Proceed with testing?")
        
    def run_single_model_test(self, model_info, datasets, output_dir, model_idx, total_models):
        """Run test for a single model"""
        variant = model_info['variant']
        checkpoint_info = model_info.get('selected_checkpoint', model_info['default_checkpoint'])
        
        print(f"\n{'='*80}")
        print(f"🎯 TESTING MODEL [{model_idx}/{total_models}]: {variant.upper()}")
        print(f"🔧 Checkpoint: {checkpoint_info['display_name']} ({checkpoint_info['date']})")
        print(f"{'='*80}")
        
        try:
            # Load model with selected checkpoint
            print(f"🔧 Loading model: {model_info['display_name']}")
            model, loaded_checkpoint = setup_model_and_checkpoint_with_selection(
                model_info, checkpoint_info
            )
            model.netG.eval()
            
            results = {}
            
            # Add checkpoint info to results
            results['checkpoint_info'] = {
                'type': loaded_checkpoint['type'],
                'iteration': loaded_checkpoint['iteration'],
                'display_name': loaded_checkpoint['display_name'],
                'date': loaded_checkpoint['date'],
                'path': loaded_checkpoint['path']
            }
            
            # Test original dataset
            if datasets['test_original']:
                print(f"\n📊 Testing original dataset...")
                try:
                    original_results = process_original_dataset(
                        model, datasets['original_path'], output_dir, self.device, variant
                    )
                    results['original_dataset'] = original_results
                    print(f"✅ Original dataset completed - PSNR: {original_results['metrics']['psnr']['mean']:.4f} dB")
                except Exception as e:
                    print(f"❌ Original dataset failed: {e}")
                    
            # Test clinical dataset
            if datasets['test_clinical']:
                print(f"\n🏥 Testing clinical dataset...")
                try:
                    clinical_results = process_clinical_dataset_no_masks(
                        model, datasets['clinical_path'], output_dir, self.device, variant
                    )
                    results['clinical_dataset_no_masks'] = clinical_results
                    print(f"✅ Clinical dataset completed - {clinical_results['total_samples']} slices processed")
                except Exception as e:
                    print(f"❌ Clinical dataset failed: {e}")
            # Test clinical artifact-only dataset
            if datasets['test_clinical_artifact_only']:
                print(f"\n🧪 Testing clinical artifact-only dataset...")
                try:
                    clinical_artifact_results = process_clinical_artifact_only_dataset(
                        model, datasets['clinical_artifact_only_path'], output_dir, self.device, variant
                    )
                    results['clinical_artifact_only_dataset'] = clinical_artifact_results
                    print(f"✅ Clinical artifact-only dataset completed - {clinical_artifact_results['total_samples']} images processed")
                except Exception as e:
                    print(f"❌ Clinical artifact-only dataset failed: {e}")
                    
            # Clear GPU memory
            del model
            torch.cuda.empty_cache()
            
            print(f"\n✅ COMPLETED {variant.upper()}")
            return results
            
        except Exception as e:
            print(f"❌ Model {variant} failed completely: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            return None
            
    def run_testing_session(self, models, datasets, output_info):
        """Run the main testing session"""
        output_dir = output_info['path']
        
        # Use resume models if available, otherwise use selected models
        if output_info['mode'] == 'resume' and output_info['models']:
            models_to_test = output_info['models']
            print(f"🔄 Resuming with {len(models_to_test)} remaining models")
        else:
            models_to_test = models
            # Reset resume status for new/continue modes
            for model in models_to_test:
                model['resume_status'] = None
                
        all_results = {}
        
        print(f"\n🚀 Starting testing session...")
        print(f"📂 Output directory: {output_dir}")
        
        # Process each model
        for model_idx, model_info in enumerate(models_to_test, 1):
            variant = model_info['variant']
            
            # Skip if resuming and already completed
            resume_status = model_info.get('resume_status')
            if resume_status:
                skip_original = resume_status.get('original_complete', False) and datasets['test_original']
                skip_clinical = resume_status.get('clinical_complete', False) and datasets['test_clinical']
                skip_clinical_artifact = resume_status.get('clinical_artifact_complete', False) and datasets['test_clinical_artifact_only']
                
                if skip_original and skip_clinical and skip_clinical_artifact:
                    print(f"\n✅ [{model_idx}/{len(models_to_test)}] {variant} already completed, skipping...")
                    continue
                    
            results = self.run_single_model_test(
                model_info, datasets, output_dir, model_idx, len(models_to_test)
            )
            
            if results:
                all_results[variant] = results
                
        # Create comprehensive analysis if we have results
        if all_results:
            print(f"\n{'='*80}")
            print(f"📊 CREATING COMPREHENSIVE ANALYSIS")
            print(f"{'='*80}")
            
            try:
                comprehensive_analysis = create_comprehensive_analysis(all_results, output_dir)
                print(f"✅ Analysis created successfully!")
                
                # Print final summary
                print(f"\n🎉 TESTING SESSION COMPLETED!")
                print(f"{'='*80}")
                print(f"📈 Models tested: {len(all_results)}")
                print(f"📂 Results saved to: {os.path.basename(output_dir)}")
                
                # Show rankings if available
                if 'original_dataset' in comprehensive_analysis.get('comparison_analysis', {}):
                    original_analysis = comprehensive_analysis['comparison_analysis']['original_dataset']
                    print(f"\n🏆 ORIGINAL DATASET RANKINGS:")
                    for i, (variant, psnr) in enumerate(original_analysis['psnr_ranking'].items(), 1):
                        ssim = original_analysis['ssim_ranking'][variant]
                        print(f"   {i}. {variant:<15} | PSNR: {psnr:.4f} dB | SSIM: {ssim:.6f}")
                        
            except Exception as e:
                print(f"⚠️  Could not create comprehensive analysis: {e}")
                
        else:
            print("❌ No results generated! All models failed to process.")
            
        print(f"\n📁 Check results in: {output_dir}")
        
    def configure_results_settings(self):
        """Configure how many results to save and what types"""
        print("\n⚙️  CONFIGURE RESULTS & OUTPUT SETTINGS")
        print("="*50)
        
        # Quick preset selection
        preset_options = [
            "Minimal (fast, few images)",
            "Standard (balanced)",
            "Comprehensive (detailed, many images)",
            "Fast (very minimal output)",
            "Custom configuration"
        ]
        
        self.print_menu(preset_options, "Choose results configuration")
        choice = self.get_user_choice(len(preset_options))
        
        if choice == 0:
            return False
        elif choice == 1:  # Minimal
            self.results_config.get_preset('minimal')
            self.show_config_summary()
        elif choice == 2:  # Standard
            self.results_config.get_preset('standard')
            self.show_config_summary()
        elif choice == 3:  # Comprehensive
            self.results_config.get_preset('comprehensive')
            self.show_config_summary()
        elif choice == 4:  # Fast
            self.results_config.get_preset('fast')
            self.show_config_summary()
        elif choice == 5:  # Custom
            return self.configure_custom_results()
            
        # Ask if user wants to modify the preset
        if choice in [1, 2, 3, 4]:
            if self.get_yes_no("📝 Modify these settings?"):
                return self.configure_custom_results()
        
        return True
        
    def configure_custom_results(self):
        """Configure custom results settings"""
        print("\n📋 CUSTOM RESULTS CONFIGURATION")
        print("-" * 40)
        
        while True:
            config_options = [
                f"Image saving frequency (currently every {self.results_config.save_every_nth})",
                f"Best/worst samples to save ({self.results_config.save_best_samples}/{self.results_config.save_worst_samples})",
                f"Maximum images limits (comparisons: {self.results_config.max_comparisons})",
                f"Clinical dataset settings (save every {self.results_config.clinical_save_frequency})",
                f"Image format & quality ({self.results_config.image_format}, DPI: {self.results_config.image_dpi})",
                f"Analysis options (metrics plots: {self.results_config.create_metrics_plots})",
                "Performance & memory settings",
                "View current configuration",
                "Save and continue"
            ]
            
            self.print_menu(config_options, "What would you like to configure?")
            choice = self.get_user_choice(len(config_options))
            
            if choice == 0:
                return False
            elif choice == 1:
                self.configure_image_frequency()
            elif choice == 2:
                self.configure_sample_limits()
            elif choice == 3:
                self.configure_image_limits()
            elif choice == 4:
                self.configure_clinical_settings()
            elif choice == 5:
                self.configure_image_format()
            elif choice == 6:
                self.configure_analysis_options()
            elif choice == 7:
                self.configure_performance_settings()
            elif choice == 8:
                self.show_config_summary()
            elif choice == 9:
                return True
                
    def configure_image_frequency(self):
        """Configure how often to save images"""
        print("\n📷 IMAGE SAVING FREQUENCY")
        print("-" * 30)
        
        try:
            current = self.results_config.save_every_nth
            new_freq = input(f"Save comparison images every N samples [{current}]: ").strip()
            if new_freq:
                self.results_config.save_every_nth = max(1, int(new_freq))
            
            current = self.results_config.preprocessing_every_nth
            new_freq = input(f"Save preprocessing visualizations every N samples [{current}]: ").strip()
            if new_freq:
                self.results_config.preprocessing_every_nth = max(1, int(new_freq))
                
            current = self.results_config.detailed_every_nth
            new_freq = input(f"Save detailed analysis every N samples [{current}]: ").strip()
            if new_freq:
                self.results_config.detailed_every_nth = max(1, int(new_freq))
                
        except ValueError:
            print("❌ Please enter valid numbers")
            
    def configure_sample_limits(self):
        """Configure best/worst sample limits"""
        print("\n🏆 BEST/WORST SAMPLE LIMITS")
        print("-" * 35)
        
        try:
            current = self.results_config.save_best_samples
            new_limit = input(f"Number of best PSNR/SSIM samples to save [{current}]: ").strip()
            if new_limit:
                self.results_config.save_best_samples = max(0, int(new_limit))
            
            current = self.results_config.save_worst_samples
            new_limit = input(f"Number of worst PSNR/SSIM samples to save [{current}]: ").strip()
            if new_limit:
                self.results_config.save_worst_samples = max(0, int(new_limit))
                
        except ValueError:
            print("❌ Please enter valid numbers")
            
    def configure_image_limits(self):
        """Configure maximum image limits"""
        print("\n🖼️  MAXIMUM IMAGE LIMITS")
        print("-" * 30)
        
        try:
            current = self.results_config.max_comparisons
            new_limit = input(f"Maximum comparison images [{current}]: ").strip()
            if new_limit:
                self.results_config.max_comparisons = max(1, int(new_limit))
            
            current = self.results_config.max_preprocessing_vis
            new_limit = input(f"Maximum preprocessing visualizations [{current}]: ").strip()
            if new_limit:
                self.results_config.max_preprocessing_vis = max(1, int(new_limit))
                
            current = self.results_config.max_detailed_analysis
            new_limit = input(f"Maximum detailed analysis images [{current}]: ").strip()
            if new_limit:
                self.results_config.max_detailed_analysis = max(1, int(new_limit))
                
        except ValueError:
            print("❌ Please enter valid numbers")
            
    def configure_clinical_settings(self):
        """Configure clinical dataset specific settings"""
        print("\n🏥 CLINICAL DATASET SETTINGS")
        print("-" * 35)
        
        try:
            current = self.results_config.clinical_save_frequency
            new_freq = input(f"Save clinical results every N slices [{current}]: ").strip()
            if new_freq:
                self.results_config.clinical_save_frequency = max(1, int(new_freq))
            
            current = self.results_config.clinical_sample_limit or "unlimited"
            new_limit = input(f"Limit clinical processing to N slices [{current}]: ").strip()
            if new_limit:
                if new_limit.lower() in ['none', 'unlimited', '0']:
                    self.results_config.clinical_sample_limit = None
                else:
                    self.results_config.clinical_sample_limit = max(1, int(new_limit))
                    
        except ValueError:
            print("❌ Please enter valid numbers")
            
    def configure_image_format(self):
        """Configure image format and quality settings"""
        print("\n🎨 IMAGE FORMAT & QUALITY")
        print("-" * 30)
        
        format_options = ["png", "jpg", "tiff"]
        print(f"Current format: {self.results_config.image_format}")
        for i, fmt in enumerate(format_options, 1):
            print(f"  {i}. {fmt.upper()}")
        
        try:
            choice = input("Choose format (1-3) or press Enter to keep current: ").strip()
            if choice:
                choice_int = int(choice)
                if 1 <= choice_int <= 3:
                    self.results_config.image_format = format_options[choice_int - 1]
            
            if self.results_config.image_format == 'jpg':
                current = self.results_config.image_quality
                new_quality = input(f"JPEG quality (1-100) [{current}]: ").strip()
                if new_quality:
                    self.results_config.image_quality = max(1, min(100, int(new_quality)))
            
            current = self.results_config.image_dpi
            new_dpi = input(f"Image DPI [{current}]: ").strip()
            if new_dpi:
                self.results_config.image_dpi = max(72, int(new_dpi))
                
        except ValueError:
            print("❌ Please enter valid numbers")
            
    def configure_analysis_options(self):
        """Configure analysis and metrics options"""
        print("\n📊 ANALYSIS OPTIONS")
        print("-" * 25)
        
        self.results_config.create_metrics_plots = self.get_yes_no(
            f"Create metrics distribution plots? (currently: {self.results_config.create_metrics_plots})"
        )
        
        self.results_config.save_individual_metrics = self.get_yes_no(
            f"Save individual sample metrics? (currently: {self.results_config.save_individual_metrics})"
        )
        
        self.results_config.create_comprehensive_analysis = self.get_yes_no(
            f"Create comprehensive analysis? (currently: {self.results_config.create_comprehensive_analysis})"
        )
        
        self.results_config.compress_results = self.get_yes_no(
            f"Compress results after completion? (currently: {self.results_config.compress_results})"
        )
        
    def configure_performance_settings(self):
        """Configure performance and memory settings"""
        print("\n⚡ PERFORMANCE & MEMORY")
        print("-" * 30)
        
        self.results_config.batch_process = self.get_yes_no(
            f"Use batch processing to save memory? (currently: {self.results_config.batch_process})"
        )
        
        try:
            current = self.results_config.clear_cache_frequency
            new_freq = input(f"Clear CUDA cache every N samples [{current}]: ").strip()
            if new_freq:
                self.results_config.clear_cache_frequency = max(1, int(new_freq))
        except ValueError:
            print("❌ Please enter valid numbers")
            
        self.results_config.verbose_logging = self.get_yes_no(
            f"Enable verbose logging? (currently: {self.results_config.verbose_logging})"
        )
        
        self.results_config.save_processing_logs = self.get_yes_no(
            f"Save detailed processing logs? (currently: {self.results_config.save_processing_logs})"
        )
        
    def show_config_summary(self):
        """Show current configuration summary"""
        print(f"\n📋 CURRENT RESULTS CONFIGURATION")
        print("="*45)
        print(f"💾 Image Saving:")
        print(f"   • Comparison images: every {self.results_config.save_every_nth} samples")
        print(f"   • Preprocessing vis: every {self.results_config.preprocessing_every_nth} samples")
        print(f"   • Detailed analysis: every {self.results_config.detailed_every_nth} samples")
        
        print(f"\n🏆 Sample Limits:")
        print(f"   • Best samples: {self.results_config.save_best_samples}")
        print(f"   • Worst samples: {self.results_config.save_worst_samples}")
        
        print(f"\n📊 Maximum Outputs:")
        print(f"   • Max comparisons: {self.results_config.max_comparisons}")
        print(f"   • Max preprocessing: {self.results_config.max_preprocessing_vis}")
        print(f"   • Max detailed: {self.results_config.max_detailed_analysis}")
        
        print(f"\n🏥 Clinical Settings:")
        print(f"   • Save frequency: every {self.results_config.clinical_save_frequency} slices")
        limit_text = str(self.results_config.clinical_sample_limit) if self.results_config.clinical_sample_limit else "unlimited"
        print(f"   • Sample limit: {limit_text}")
        
        print(f"\n🎨 Format & Quality:")
        print(f"   • Format: {self.results_config.image_format.upper()}")
        if self.results_config.image_format == 'jpg':
            print(f"   • Quality: {self.results_config.image_quality}%")
        print(f"   • DPI: {self.results_config.image_dpi}")
        
        print(f"\n📈 Analysis:")
        print(f"   • Metrics plots: {'✓' if self.results_config.create_metrics_plots else '✗'}")
        print(f"   • Individual metrics: {'✓' if self.results_config.save_individual_metrics else '✗'}")
        print(f"   • Comprehensive analysis: {'✓' if self.results_config.create_comprehensive_analysis else '✗'}")
        print("="*45)
        
    def run(self):
        """Main CLI loop"""
        self.print_header()
        
        try:
            while True:
                # Step 1: Select models
                models = self.discover_and_select_models()
                if not models:
                    print("👋 Goodbye!")
                    break
                    
                # Step 2: Select datasets
                datasets = self.select_datasets()
                if not datasets:
                    continue
                    
                # Step 3: Manage output folder
                output_info = self.manage_output_folder()
                if not output_info:
                    continue
                    
                # Step 4: Configure results settings
                if not self.configure_results_settings():
                    continue
                
                # Step 5: Show summary and confirm
                if not self.show_test_summary(models, datasets, output_info):
                    continue
                    
                # Step 6: Run testing session
                self.run_testing_session(models, datasets, output_info)
                
                # Step 7: Ask if user wants to run another session
                if not self.get_yes_no("\n🔄 Run another testing session?"):
                    print("👋 Goodbye!")
                    break
                    
        except KeyboardInterrupt:
            print("\n\n⏹️  Testing interrupted by user. Goodbye!")
        except Exception as e:
            print(f"\n❌ Fatal error: {e}")
            traceback.print_exc()

def main():
    """Main entry point"""
    cli = InteractiveCLI()
    cli.run()

if __name__ == '__main__':
    main()
