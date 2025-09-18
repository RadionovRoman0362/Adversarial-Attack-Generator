import yaml
from torch.utils.tensorboard import SummaryWriter
from typing import List, Dict, Any
import io
import matplotlib.pyplot as plt
import pandas as pd
import torch


class TensorBoardLogger:
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)

    def log_population_distribution_plot(self, population: List[Dict[str, Any]], generation: int):
        if not population:
            return

        def get_gradient_components(ind):
            chain = []
            curr = ind.get('gradient')
            while curr:
                chain.append(curr['name'])
                curr = curr.get('wrapped')
            return chain

        all_grad_components = []
        for ind in population:
            all_grad_components.extend(get_gradient_components(ind))

        if not all_grad_components:
            return

        component_counts = pd.Series(all_grad_components).value_counts()

        fig, ax = plt.subplots(figsize=(10, 6))
        component_counts.sort_values(ascending=True).plot(kind='barh', ax=ax)
        ax.set_title(f'Распределение Компонентов Градиента в Поколении {generation}')
        ax.set_xlabel('Количество в популяции')
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        image = torch.from_numpy(plt.imread(buf, format='png').transpose((2, 0, 1)))
        self.writer.add_image('Gradient_Distribution/bar_chart', image, global_step=generation)

        plt.close(fig)

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