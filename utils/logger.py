import rich
from rich.table import Table

class Logger():
    def __init__(self):
        self.console = rich.get_console()
        self.console = self.console.__class__(log_time=False)

    def log(self, message: str):
        self.console.log(message)

    def print(self, message: str):
        self.console.print(message)
        
    def console_banner(self):
        self.console.clear()
        banner = [
            "\n",
            "    ___       ___       ___       ___            ___       ___       ___            ___       ___       ___       ___    ",
            "   /\__\     /\  \     /\__\     /\  \          /\  \     /\  \     /\__\          /\  \     /\  \     /\  \     /\__\   ",
            "  /:| _|_   /::\  \   |::L__L    \:\  \        /::\  \   /::\  \   /:| _|_        /::\  \   /::\  \    \:\  \   /:/ _/_  ",
            " /::|/\__\ /::\:\__\ /::::\__\   /::\__\      /:/\:\__\ /::\:\__\ /::|/\__\      /:/\:\__\ /:/\:\__\   /::\__\ |::L/\__\ ",
            " \/|::/  / \:\:\/  / \;::;/__/  /:/\/__/      \:\:\/__/ \:\:\/  / \/|::/  /      \:\ \/__/ \:\ \/__/  /:/\/__/ |::::/  / ",
            "   |:/  /   \:\/  /   |::|__|   \/__/          \::/  /   \:\/  /    |:/  /        \:\__\    \:\__\    \/__/     L;;/__/  ",
            "   \/__/     \/__/     \/__/                    \/__/     \/__/     \/__/          \/__/     \/__/            ",
            "\n",
            "Pipe-Action: Pipeline for segmenting specific people from video containing multiple individuals.",
            "\n",
        ]
        for line in banner:
            self.console.print(line, style="bold green")
            
    def clear(self):
        self.console.clear()
            
    def console_args(self, args):
        table = Table(show_header=True, show_footer=False)
        table.add_column("Argument", width=17)
        table.add_column("Value", width=40)
        for arg, value in args.items():
            table.add_row(arg, str(value))
        self.console.print(table)

    def table(self, data: list):
        table = Table(show_header=True, header_style="bold magenta")
        for row in data:
            table.add_row(*[str(item) for item in row])
        self.console.print(table)
