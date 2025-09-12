# advgen/utils/visualization.py
import yaml
from torch.utils.tensorboard import SummaryWriter
from typing import List, Dict, Any


class TensorBoardLogger:
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)

    def log_generation_stats(self, population: List[Dict[str, Any]], fitness_scores: List[float], generation: int):
        """Логирует скаляры и гистограммы для поколения."""
        if not fitness_scores: return

        best_fitness = max(fitness_scores)
        avg_fitness = sum(fitness_scores) / len(fitness_scores)

        self.writer.add_scalar('ASR/best', best_fitness, generation)
        self.writer.add_scalar('ASR/average', avg_fitness, generation)

        component_types = population[0].keys()
        for comp_type in component_types:
            component_names = [p[comp_type]['name'] for p in population]

            name_counts = {name: component_names.count(name) for name in set(component_names)}

            text_summary = f"Distribution for {comp_type}:\n\n" + "\n".join(
                [f"{k}: {v}" for k, v in name_counts.items()])
            self.writer.add_text(f'Distributions/{comp_type}', text_summary, generation)

    def log_best_individual(self, best_individual: Dict[str, Any], best_fitness: float, generation: int):
        """Логирует конфигурацию лучшего индивида."""
        config_text = f"Best ASR: {best_fitness:.4f}\n\n" + yaml.dump(best_individual, indent=2)
        self.writer.add_text('Best_Config_Of_Generation', config_text, generation)

    def close(self):
        self.writer.close()